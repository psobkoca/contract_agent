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
    # 1. Initialize Dual Logging
    setup_dual_logging()
    logger.info(f"Starting contract review pipeline execution. Run ID: {run_id}")
    
    # 2. Retrain Classifier if requested
    if force_retrain:
        logger.info("Retraining Clause Classifier (force_retrain=True)...")
        classifier = ClauseClassifier()
        classifier.train_model(force_retrain=True)
        
    # 3. Build/Load RAG Vector Store
    logger.info("Initializing RAG Engine...")
    rag = RAGEngine()
    rag.build_or_load_vector_store(force_rebuild=force_rebuild)
    
    # 4. Initialize Scorer and Agent
    scorer = RiskScorer(rag_engine=rag)
    agent = ContractAgent(llm_client=None, rag_engine=rag, risk_scorer=scorer)
    
    # 5. Determine which contracts to process
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
        
    all_reviewed_clauses = []
    scorecard_data = []
    
    # Import parser helper
    from contract_parser import parse_contract
    
    # 6. Process each contract
    for filename in contract_files:
        filepath = os.path.join(contracts_dir, filename)
        contract_id, _ = os.path.splitext(filename)
        logger.info(f"Orchestrating analysis for contract: {filename}")
        
        try:
            # Parse and classify
            clauses = parse_contract(filepath)
            if not clauses:
                logger.warning(f"No clauses parsed from contract {contract_id}")
                continue
                
            # Score each clause
            clause_scores = []
            scored_clauses_tier = []
            for clause in clauses:
                precedents = []
                if clause.risk_flag in ["HIGH_RISK", "REVIEW_REQUIRED"]:
                    precedents = rag.hybrid_search(clause.raw_text, rerank=True, top_n=3)
                
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
                    
                scored_clauses_tier.append({
                    "clause": clause,
                    "score": score,
                    "tier": tier
                })
                
            # Calculate overall contract risk score & tier
            contract_score = sum(clause_scores) / len(clause_scores) if clause_scores else 0.0
            contract_tier = scorer.classify_tier(contract_score)
            
            # Extract contract metadata from the first clause
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
            
            # Review and generate redlines for only CRITICAL and HIGH clauses
            filtered_clauses = [c for c in scored_clauses_tier if c["tier"] in ["CRITICAL", "HIGH"]]
            sorted_clauses = sorted(filtered_clauses, key=lambda x: x["score"], reverse=True)
            
            # Cap at max_clauses_per_run
            max_clauses = agent.llm_client.token_limit # Wait, config.agent.max_clauses_per_run is standard
            from config import config
            max_clauses = config.agent.max_clauses_per_run
            selected_clauses = sorted_clauses[:max_clauses]
            
            logger.info(f"Contract {contract_id}: selected top-{len(selected_clauses)} CRITICAL/HIGH clauses for agent review.")
            
            for item in selected_clauses:
                clause = item["clause"]
                score = item["score"]
                tier = item["tier"]
                
                # Run ReAct loop review
                logger.info(f"Reviewing {tier} clause {clause.clause_id}...")
                res = agent.review_clause(clause)
                
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
                    "fallback_mode": res.fallback_mode
                }
                all_reviewed_clauses.append(redline_entry)
                
                # Log critical alerts to critical_clauses.log
                if tier == "CRITICAL":
                    logger.bind(is_critical=True).warning(
                        f"[CRITICAL RISK ALERT] Clause {clause.clause_id} in {contract_id} "
                        f"has a risk score of {score:.4f} ({clause.clause_type}). "
                        f"Priority: {res.negotiation_priority}. "
                        f"Original Text: {clause.raw_text[:120]}..."
                    )
                    
        except Exception as e:
            logger.error(f"Failed to process contract {contract_id}: {e}")
            
    # 7. Generate Reports
    reporter = Reporter(run_id=run_id)
    reporter.generate_all_reports(all_reviewed_clauses, scorecard_data)
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
