import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from loguru import logger

from clause_classifier import ClauseClassifier
from rag_engine import RAGEngine
from risk_scorer import RiskScorer
from agent import ContractAgent
from reporter import setup_dual_logging, Reporter

def run_pipeline(
    contract_path: Optional[str] = None,
    contract_id_filter: Optional[str] = None,
    force_retrain: bool = False,
    force_rebuild: bool = False,
    run_id: str = "latest"
) -> None:
    # Initialize Dual Logging
    setup_dual_logging()
    logger.info(f"Starting contract review pipeline execution. Run ID: {run_id}")
    
    # Determine which contracts to process
    contracts_dir = "contracts"
    if contract_path:
        if not os.path.exists(contract_path):
            logger.error(f"Specified contract path does not exist: {contract_path}")
            sys.exit(1)
        contract_files = [os.path.basename(contract_path)]
        contracts_dir = os.path.dirname(contract_path) or "."
    else:
        # Scan contracts directory
        if not os.path.exists(contracts_dir):
            logger.error(f"Contracts directory not found: {contracts_dir}")
            sys.exit(1)
        all_files = os.listdir(contracts_dir)
        contract_files = [
            f for f in all_files if f.endswith(".pdf") or f.endswith(".docx")
        ]
        
    # Filter by contract_id if requested
    if contract_id_filter:
        contract_files = [
            f for f in contract_files if contract_id_filter in f
        ]
        logger.info(f"Filtered contract list by ID '{contract_id_filter}'. Processing {len(contract_files)} contracts.")
        
    if not contract_files:
        logger.warning("No contracts selected for processing.")
        return

    # Import parser helper
    from contract_parser import parse_contract

    # Step 1: Parse contract file (PDF or DOCX); segment into clauses; enrich with contract metadata.
    logger.info("Step 1: Parsing contract files, segmenting into clauses, and enriching with metadata...")
    contracts_clauses = {}
    for filename in contract_files:
        filepath = os.path.join(contracts_dir, filename)
        contract_id, _ = os.path.splitext(filename)
        logger.info(f"Parsing contract {filename}...")
        try:
            clauses = parse_contract(filepath, classify=False)
            if not clauses:
                logger.warning(f"No clauses parsed from contract {contract_id}")
                continue
            contracts_clauses[contract_id] = clauses
        except Exception as e:
            logger.error(f"Failed to parse contract {contract_id}: {e}")

    if not contracts_clauses:
        logger.warning("No contract clauses successfully parsed. Exiting pipeline.")
        return

    # Step 2: Load or train clause classifier; predict clause_type and confidence for each clause.
    logger.info("Step 2: Loading or training clause classifier and predicting clause type & confidence...")
    classifier = ClauseClassifier()
    classifier.init_classifier(force_retrain=force_retrain)
    for contract_id, clauses in contracts_clauses.items():
        logger.info(f"Predicting clause types for contract: {contract_id}")
        for clause in clauses:
            pred_class, confidence = classifier.predict_sklearn(clause.raw_text)
            clause.clause_type = pred_class
            clause.confidence = confidence

    # Step 3: Apply zero-shot NLI fallback for clauses below confidence threshold; assign initial risk_flag.
    logger.info("Step 3: Applying zero-shot NLI fallback for low-confidence clauses and assigning initial risk flags...")
    from config import config
    confidence_threshold = config.classifier.confidence_threshold
    for contract_id, clauses in contracts_clauses.items():
        for clause in clauses:
            if clause.confidence < confidence_threshold:
                nli_class, nli_confidence = classifier.predict_nli_fallback(clause.raw_text)
                clause.clause_type = nli_class
                clause.confidence = nli_confidence
            clause.risk_flag = classifier.get_risk_flag(clause.clause_type, clause.confidence)

    # Step 4: Build or reload ChromaDB vector store from legal_precedents/ directory.
    logger.info("Step 4: Building or reloading ChromaDB vector store...")
    rag = RAGEngine()
    rag.build_or_load_vector_store(force_rebuild=force_rebuild)

    # Step 5: For each HIGH_RISK or REVIEW_REQUIRED clause, execute hybrid RAG retrieval; rerank top-3 passages.
    logger.info("Step 5: Executing hybrid RAG retrieval and reranking for HIGH_RISK/REVIEW_REQUIRED clauses...")
    clause_precedents = {}
    for contract_id, clauses in contracts_clauses.items():
        for clause in clauses:
            if clause.risk_flag in ["HIGH_RISK", "REVIEW_REQUIRED"]:
                precedents = rag.hybrid_search(clause.raw_text, rerank=True, top_n=3)
                clause_precedents[clause.clause_id] = precedents
            else:
                clause_precedents[clause.clause_id] = []

    # Step 6: Compute four-factor Risk Score per clause; classify into CRITICAL/HIGH/MEDIUM/LOW tiers; export risk_scorecard.csv.
    logger.info("Step 6: Computing risk scores, classifying risk tiers, and exporting risk_scorecard.csv...")
    scorer = RiskScorer(rag_engine=rag)
    scorecard_data = []
    scored_contracts_clauses = {}
    
    for contract_id, clauses in contracts_clauses.items():
        clause_scores = []
        scored_clauses_tier = []
        for clause in clauses:
            precedents = clause_precedents.get(clause.clause_id, [])
            clause_res = scorer.score_clause(clause, precedents)
            score = clause_res["final_score"]
            clause_scores.append(score)
            
            # Determine clause risk tier
            if score >= 0.70:
                tier = "CRITICAL"
            elif score >= 0.50:
                tier = "HIGH"
            elif score >= 0.35:
                tier = "MEDIUM"
            else:
                tier = "LOW"
                
            # Override tier to HIGH if LLM review is required for this clause type regardless of confidence
            c_type = clause.clause_type
            if c_type in scorer.clause_type_weights and scorer.clause_type_weights[c_type]["review_required"]:
                if tier not in ["CRITICAL", "HIGH"]:
                    tier = "HIGH"
                
            scored_clauses_tier.append({
                "clause": clause,
                "score": score,
                "tier": tier
            })
            
        scored_contracts_clauses[contract_id] = scored_clauses_tier
        
        # Calculate overall contract risk score & tier
        contract_score = sum(clause_scores) / len(clause_scores) if clause_scores else 0.0
        contract_tier = scorer.classify_tier(contract_score)
        
        first_clause = clauses[0]
        scorecard_data.append({
            "contract_id": contract_id,
            "contract_type": first_clause.contract_type or "Unknown",
            "counterparty_name": first_clause.counterparty_name or "Unknown",
            "governing_law": first_clause.governing_law_jurisdiction or "Unknown",
            "contract_value_usd": first_clause.contract_value_usd or 0.0,
            "risk_score": contract_score,
            "risk_tier": contract_tier
        })
        
    # Export risk_scorecard.csv immediately
    reporter = Reporter(run_id=run_id)
    reporter.generate_csv_scorecard(scorecard_data)

    # Step 7: Sort clauses by Risk Score descending; select top CRITICAL/HIGH tier clauses up to max_clauses_per_run.
    logger.info("Step 7: Sorting clauses and selecting top CRITICAL/HIGH clauses up to max_clauses_per_run...")
    all_scored_clauses = []
    for contract_id, scored_items in scored_contracts_clauses.items():
        all_scored_clauses.extend(scored_items)
        
    filtered_clauses = [item for item in all_scored_clauses if item["tier"] in ["CRITICAL", "HIGH"]]
    sorted_clauses = sorted(filtered_clauses, key=lambda x: x["score"], reverse=True)
    
    max_clauses = config.agent.max_clauses_per_run
    selected_clauses = sorted_clauses[:max_clauses]
    logger.info(f"Selected top {len(selected_clauses)} CRITICAL/HIGH clauses for review (limit: {max_clauses}).")

    # Step 8: Build structured LLM prompt per clause: clause text + classifier output + top-3 passages + risk scores + org_profile context.
    # Step 9: Call Claude API; parse Pydantic LLMResponse; if tool_call for precedent_lookup or jurisdiction_lookup, execute and append (max 3 iterations per clause); validate mandatory disclaimer.
    logger.info("Steps 8 & 9: Running LLM Agent review loop for selected clauses...")
    agent = ContractAgent(llm_client=None, rag_engine=rag, risk_scorer=scorer)
    all_reviewed_clauses = []
    contract_redlines = {cid: [] for cid in contracts_clauses.keys()}
    
    for item in selected_clauses:
        clause = item["clause"]
        score = item["score"]
        tier = item["tier"]
        contract_id = clause.clause_id.split("_CLS_")[0] if "_CLS_" in clause.clause_id else "Unknown"
        
        logger.info(f"Reviewing {tier} clause {clause.clause_id} from contract {contract_id} (Score={score:.4f})...")
        try:
            res = agent.review_clause(clause)
            
            # Fetch precedent citations
            precedents = clause_precedents.get(clause.clause_id, [])
            precedent_citations = [
                f"{p['metadata'].get('source_file', 'Unknown')} - {p['metadata'].get('title', 'Unknown')}"
                for p in precedents
            ]
            
            redline_entry = {
                "clause_id": clause.clause_id,
                "section_number": clause.section_number,
                "raw_text": clause.raw_text,
                "clause_type": clause.clause_type,
                "risk_score": score,
                "risk_tier": tier,
                "original_clause_summary": res.original_clause_summary,
                "redlined_clause": res.redlined_clause,
                "redline_rationale": res.redline_rationale,
                "negotiation_priority": res.negotiation_priority,
                "walk_away_trigger": res.walk_away_trigger,
                "confidence_score": res.confidence_score,
                "legal_disclaimer": res.legal_disclaimer,
                "fallback_mode": res.fallback_mode,
                "loop_count": res.loop_count,
                "latency_ms": res.latency_ms,
                "precedent_citations": precedent_citations
            }
            all_reviewed_clauses.append(redline_entry)
            if contract_id in contract_redlines:
                contract_redlines[contract_id].append(redline_entry)
            
            # Log critical alerts to critical_clauses.log
            if tier == "CRITICAL":
                logger.bind(is_critical=True).warning(
                    f"[CRITICAL RISK ALERT] Clause {clause.clause_id} in {contract_id} "
                    f"has a risk score of {score:.4f} ({clause.clause_type}). "
                    f"Priority: {res.negotiation_priority}. "
                    f"Original Text: {clause.raw_text[:120]}..."
                )
        except Exception as e:
            logger.error(f"Failed to review clause {clause.clause_id}: {e}")

    # Build FR-22 compliant JSON log payload
    import datetime
    run_timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    contract_review_logs = []
    for contract_id, clauses in contracts_clauses.items():
        first_clause = clauses[0]
        counterparty_name = first_clause.counterparty_name or "Unknown"
        
        redlines = contract_redlines.get(contract_id, [])
        clauses_reviewed_by_llm = len(redlines)
        
        risk_tier_distribution = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        scored_items = scored_contracts_clauses.get(contract_id, [])
        for item in scored_items:
            tier = item["tier"]
            if tier == "LOW_RISK":
                tier = "LOW"
            if tier in risk_tier_distribution:
                risk_tier_distribution[tier] += 1
                
        loop_count = sum(r.get("loop_count", 0) for r in redlines)
        fallback_mode = any(r.get("fallback_mode", False) for r in redlines)
        latency_ms = sum(r.get("latency_ms", 0) for r in redlines)
        
        contract_summary = (
            f"Review of contract {contract_id} with counterparty {counterparty_name}. "
            f"A total of {len(clauses)} clauses were parsed, of which {clauses_reviewed_by_llm} "
            f"high or critical risk clauses were reviewed by the LLM."
        )
        
        log_data = {
            "contract_id": contract_id,
            "counterparty_name": counterparty_name,
            "run_timestamp": run_timestamp,
            "contract_summary": contract_summary,
            "total_clauses": len(clauses),
            "clauses_reviewed_by_llm": clauses_reviewed_by_llm,
            "risk_tier_distribution": risk_tier_distribution,
            "redlines": redlines,
            "loop_count": loop_count,
            "fallback_mode": fallback_mode,
            "latency_ms": latency_ms
        }
        contract_review_logs.append(log_data)
        
    json_log_payload = contract_review_logs[0] if len(contract_review_logs) == 1 else contract_review_logs

    # Step 10: Write JSON log, Markdown memo (disclaimer top+bottom), redlines.csv; print CLI summary with CRITICAL RISK ALERT if applicable.
    logger.info("Step 10: Writing JSON log, Markdown memo, redlines.csv, and displaying CLI summary...")
    reporter.generate_json_log(json_log_payload)
    reporter.generate_markdown_memo(contract_review_logs, scored_contracts_clauses, scorecard_data)
    reporter.generate_csv_redlines(all_reviewed_clauses)
    reporter.display_cli_summary(all_reviewed_clauses, scorecard_data)
    
    logger.success("Pipeline execution completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Contract Review Agent Pipeline")
    parser.add_argument("--contract", help="Path to a specific contract file (PDF or DOCX).")
    parser.add_argument("--contract_id", help="Optional contract ID filter to process only matching contracts.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of the clause classifier.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding of the RAG vector store.")
    parser.add_argument("--run_id", default="latest", help="Run identifier for report file naming.")
    
    args = parser.parse_args()
    
    from typing import Optional
    run_pipeline(
        contract_path=args.contract,
        contract_id_filter=args.contract_id,
        force_retrain=args.retrain,
        force_rebuild=args.rebuild,
        run_id=args.run_id
    )

if __name__ == "__main__":
    main()
