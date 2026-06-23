import os
import sys
import csv
import json
from typing import List, Dict, Any
from loguru import logger
from tabulate import tabulate

import prompts

def setup_dual_logging() -> None:
    """Configures loguru to log to stdout, agent.log (all), and critical_clauses.log (only critical events)."""
    # Remove existing default handlers
    logger.remove()
    
    # 1. Console Output
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    # Load dynamic configurations
    from config import config
    log_path = config.output.log_path
    log_level = config.output.log_level
    
    # 2. agent.log (All logs)
    logger.add(
        log_path,
        rotation="10 MB",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    
    # 3. critical_clauses.log (Only logs with extra is_critical=True or containing CRITICAL)
    logger.add(
        "critical_clauses.log",
        rotation="10 MB",
        level="WARNING",
        filter=lambda r: r["extra"].get("is_critical") is True or "CRITICAL RISK" in r["message"],
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    
    logger.info("Dual logging system successfully initialized.")

class Reporter:
    """Generates run execution reports: JSON logs, Markdown memos, CSV scorecards, and CLI tables."""
    
    def __init__(self, run_id: str = "run_latest"):
        self.run_id = run_id
        from config import config
        self.output_dir = config.output.report_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_all_reports(self, run_data: List[Dict[str, Any]], scorecard_data: List[Dict[str, Any]]) -> None:
        """Generates all 4 reports: JSON review log, MD memo, CSV scorecard, CSV redlines, and displays summary."""
        self.generate_json_log(run_data)
        self.generate_markdown_memo(run_data)
        self.generate_csv_scorecard(scorecard_data)
        self.generate_csv_redlines(run_data)
        self.display_cli_summary(run_data, scorecard_data)

    def generate_json_log(self, run_data: Any) -> str:
        """Saves a detailed JSON review log containing all analyzed clauses and suggestions (FR-22)."""
        filename = f"{self.run_id}.json"
        path = os.path.join(self.output_dir, filename)
        
        try:
            with open(path, mode="w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2)
            logger.success(f"JSON review log exported to {path}")
        except Exception as e:
            logger.error(f"Failed to generate JSON review log: {e}")
        return path

    def generate_markdown_memo(
        self,
        contract_review_logs: List[Dict[str, Any]],
        scored_contracts_clauses: Dict[str, Any],
        scorecard_data: List[Dict[str, Any]]
    ) -> str:
        """Generates a Markdown review memo matching the exact FR-23 format requirements."""
        filename = f"{self.run_id}.md"
        path = os.path.join(self.output_dir, filename)
        
        # Check if there are any critical risk clauses in any contract
        has_critical = any(
            any(r.get("risk_tier") == "CRITICAL" for r in log.get("redlines", []))
            for log in contract_review_logs
        )
        
        try:
            with open(path, mode="w", encoding="utf-8") as f:
                # Top Disclaimer (visibly distinct callout box)
                f.write(f"> [!IMPORTANT]\n> **MANDATORY LEGAL DISCLAIMER**:\n> {prompts.DISCLAIMER_TEXT}\n\n")
                
                if has_critical:
                    f.write("# ⚠️ CRITICAL RISK ALERT\n")
                    f.write("> **WARNING**: This contract contains one or more **CRITICAL-tier** risk clauses. "
                            "These provisions present severe legal exposure and MUST be negotiated and approved "
                            "before execution. Details are provided below.\n\n")
                            
                f.write("# Contract Review & Negotiation Memo\n\n")
                
                for log in contract_review_logs:
                    cid = log["contract_id"]
                    cp = log["counterparty_name"]
                    
                    # Find contract-level stats from scorecard_data
                    sc_info = next((s for s in scorecard_data if s["contract_id"] == cid), None)
                    overall_tier = sc_info["risk_tier"] if sc_info else "Unknown"
                    overall_score = sc_info["risk_score"] if sc_info else 0.0
                    c_type = sc_info["contract_type"] if sc_info else "Unknown"
                    gov_law = sc_info["governing_law"] if sc_info else "Unknown"
                    val_usd = sc_info["contract_value_usd"] if sc_info else 0.0
                    
                    # Get badge color
                    badge_colors = {
                        "CRITICAL": "red",
                        "HIGH": "orange",
                        "MEDIUM": "yellow",
                        "LOW_RISK": "green",
                        "LOW": "green"
                    }
                    b_color = badge_colors.get(overall_tier, "blue")
                    badge_url = f"https://img.shields.io/badge/Overall_Risk_Tier-{overall_tier}-{b_color}?style=for-the-badge"
                    
                    f.write(f"## Contract: {cid} ({cp})\n\n")
                    f.write(f"### Executive Risk Summary\n")
                    f.write(f"![Overall Risk Tier Badge]({badge_url})\n\n")
                    f.write(f"The overall risk score for this contract is **{overall_score:.4f}**, classifying it into the **{overall_tier}** tier.\n\n")
                    
                    # Contract Statistics Table
                    scored_items = scored_contracts_clauses.get(cid, [])
                    effective_date = "Unknown"
                    if scored_items:
                        effective_date = scored_items[0]["clause"].effective_date or "Unknown"
                        
                    f.write("### Contract Statistics\n\n")
                    f.write("| Metric | Value |\n")
                    f.write("| :--- | :--- |\n")
                    f.write(f"| **Contract ID** | {cid} |\n")
                    f.write(f"| **Counterparty** | {cp} |\n")
                    f.write(f"| **Contract Type** | {c_type} |\n")
                    f.write(f"| **Governing Law** | {gov_law} |\n")
                    f.write(f"| **Effective Date** | {effective_date} |\n")
                    f.write(f"| **Contract Value (USD)** | ${val_usd:,.2f} |\n")
                    f.write(f"| **Total Clauses** | {log['total_clauses']} |\n")
                    f.write(f"| **Clauses Reviewed by LLM** | {log['clauses_reviewed_by_llm']} |\n\n")
                    
                    # Risk Scorecard Summary (Top-10 clauses by score)
                    f.write("### Risk Scorecard (Top-10 Clauses)\n\n")
                    f.write("| Rank | Clause ID | Section/Header | Clause Type | Risk Score | Risk Tier |\n")
                    f.write("| :---: | :--- | :--- | :--- | :---: | :---: |\n")
                    
                    sorted_items = sorted(scored_items, key=lambda x: x["score"], reverse=True)[:10]
                    for rank, item in enumerate(sorted_items):
                        cl = item["clause"]
                        f.write(f"| {rank+1} | {cl.clause_id} | {cl.section_number or 'N/A'} | {cl.clause_type} | {item['score']:.4f} | {item['tier']} |\n")
                    f.write("\n")
                    
                    # Per-Clause Negotiation Section
                    f.write("### Per-Clause Negotiation Details\n\n")
                    redlines = log.get("redlines", [])
                    if not redlines:
                        f.write("*No HIGH or CRITICAL risk clauses were reviewed for this contract.*\n\n")
                    else:
                        for idx, r in enumerate(redlines):
                            tier_str = f"**{r['risk_tier']} RISK**"
                            f.write(f"#### {idx+1}. Clause {r['clause_id']} (Section: {r['section_number']})\n")
                            f.write(f"- **Clause Type**: {r['clause_type']}\n")
                            f.write(f"- **Risk Level**: {tier_str} (Score: {r['risk_score']:.4f})\n")
                            f.write(f"- **Negotiation Priority**: `{r['negotiation_priority']}`\n")
                            if r.get("fallback_mode"):
                                f.write("- **Mode**: `FALLBACK_MODE` ⚠️\n")
                            f.write("\n")
                            
                            f.write("##### Original Clause Excerpt:\n")
                            f.write(f"```text\n{r['raw_text']}\n```\n\n")
                            
                            f.write("##### Redlined Clause:\n")
                            f.write(f"```text\n{r['redlined_clause']}\n```\n\n")
                            
                            f.write("##### Rationale:\n")
                            f.write(f"{r['redline_rationale']}\n\n")
                            
                            f.write("##### Walk-away Trigger:\n")
                            f.write(f"{r['walk_away_trigger']}\n\n")
                            
                            f.write("##### Precedent Citations:\n")
                            citations = r.get("precedent_citations", [])
                            if citations:
                                for cit in citations:
                                    f.write(f"- {cit}\n")
                            else:
                                f.write("- No precedents cited.\n")
                            f.write("\n---\n\n")
                            
                # Bottom Disclaimer (visibly distinct callout box)
                f.write(f"> [!IMPORTANT]\n> **MANDATORY LEGAL DISCLAIMER**:\n> {prompts.DISCLAIMER_TEXT}\n")
                
            logger.success(f"Markdown review memo exported to {path}")
        except Exception as e:
            logger.error(f"Failed to generate Markdown review memo: {e}")
        return path

    def generate_csv_scorecard(self, scorecard_data: List[Dict[str, Any]]) -> str:
        """Generates the scorecard CSV containing contract-level risk scores sorted by score descending."""
        path = os.path.join(self.output_dir, "risk_scorecard.csv")
        
        # Sort by risk score descending
        sorted_scorecard = sorted(scorecard_data, key=lambda x: x["risk_score"], reverse=True)
        
        try:
            with open(path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "contract_id", "contract_type", "counterparty_name", "governing_law", 
                    "contract_value_usd", "risk_score", "risk_tier"
                ])
                writer.writeheader()
                for row in sorted_scorecard:
                    writer.writerow(row)
            logger.success(f"CSV scorecard exported to {path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV scorecard: {e}")
        return path

    def generate_csv_redlines(self, run_data: List[Dict[str, Any]]) -> str:
        """Generates redlines.csv containing clause-level suggestions and flags."""
        path = os.path.join(self.output_dir, "redlines.csv")
        
        try:
            with open(path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "contract_id", "clause_id", "original_text", "suggested_redline", 
                    "rationale", "negotiation_priority", "fallback_mode"
                ])
                writer.writeheader()
                for c in run_data:
                    # Extract contract_id from clause_id (e.g. CTR_003_CLS_006 -> CTR_003)
                    contract_id = c["clause_id"].split("_CLS_")[0] if "_CLS_" in c["clause_id"] else "Unknown"
                    writer.writerow({
                        "contract_id": contract_id,
                        "clause_id": c["clause_id"],
                        "original_text": c["raw_text"],
                        "suggested_redline": c["redlined_clause"],
                        "rationale": c["redline_rationale"],
                        "negotiation_priority": c["negotiation_priority"],
                        "fallback_mode": c.get("fallback_mode", False)
                    })
            logger.success(f"CSV redlines exported to {path}")
        except Exception as e:
            logger.error(f"Failed to generate CSV redlines: {e}")
        return path

    def display_cli_summary(self, run_data: List[Dict[str, Any]], scorecard_data: List[Dict[str, Any]]) -> None:
        """Prints clean summary tables to the console."""
        has_critical = (
            any(s.get("risk_tier") == "CRITICAL" for s in scorecard_data) or
            any(c.get("risk_tier") == "CRITICAL" for c in run_data)
        )
        
        if has_critical:
            print("\n" + "!" * 80)
            print("⚠️  CRITICAL RISK ALERT: CRITICAL RISK DETECTED  ⚠️".center(80))
            print("!" * 80)

        print("\n" + "=" * 80)
        print("CONTRACT AUDIT SUMMARY".center(80))
        print("=" * 80)
        
        # 1. Contract Scorecard Table
        scorecard_rows = []
        for s in scorecard_data:
            scorecard_rows.append([
                s["contract_id"],
                s["counterparty_name"],
                s["risk_tier"],
                f"{s['risk_score']:.4f}",
                f"${s['contract_value_usd']:,.2f}"
            ])
            
        print("\n--- CONTRACT RISK SCORECARD ---")
        print(tabulate(
            scorecard_rows,
            headers=["Contract ID", "Counterparty", "Tier", "Risk Score", "Value (USD)"],
            tablefmt="grid"
        ))
        
        # 2. Critical/High Clauses Reviewed Table
        clause_rows = []
        for c in run_data:
            contract_id = c["clause_id"].split("_CLS_")[0] if "_CLS_" in c["clause_id"] else "Unknown"
            mode = "FALLBACK" if c.get("fallback_mode") is True else "LLM"
            clause_rows.append([
                c["clause_id"],
                contract_id,
                c["clause_type"],
                c["risk_tier"],
                f"{c['risk_score']:.4f}",
                c["negotiation_priority"],
                mode
            ])
            
        print("\n--- CRITICAL / HIGH CLAUSES REVIEW SUGGESTIONS ---")
        if clause_rows:
            print(tabulate(
                clause_rows,
                headers=["Clause ID", "Contract", "Type", "Tier", "Score", "Priority", "Mode"],
                tablefmt="grid"
            ))
        else:
            print("No CRITICAL or HIGH risk clauses were detected or reviewed.")
            
        print("\n" + "=" * 80 + "\n")
