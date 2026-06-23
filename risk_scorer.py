import os
import csv
import math
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

from config import config
from models import Clause
from contract_parser import parse_contract
from rag_engine import RAGEngine

class RiskScorer:
    """Assess and score contract clause-level and contract-level legal risks."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        rag_engine: Optional[RAGEngine] = None,
        jurisdiction_risk_path: str = "jurisdiction_risk.csv"
    ):
        # Configurable weights (defaulting to balanced 4-factor allocation)
        self.weights = weights or config.risk_scoring.weights or {
            "type": 0.35,
            "one_sidedness": 0.30,
            "market_deviation": 0.25,
            "jurisdiction": 0.10
        }
        
        # Verify and normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if not math.isclose(total_w, 1.0, rel_tol=1e-5):
            logger.warning(f"Weights do not sum to 1.0 (sum={total_w}). Normalizing weights...")
            self.weights = {k: v / total_w for k, v in self.weights.items()}
            
        self.rag_engine = rag_engine
        self.embedding_model = rag_engine.embedding_model if rag_engine else None
        
        # Load jurisdiction risk map
        self.jurisdiction_risk = {}
        self._load_jurisdiction_risk(jurisdiction_risk_path)

    def _load_jurisdiction_risk(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Jurisdiction risk file not found at {path}. Using fallback default risk scores.")
            self.jurisdiction_risk = {"other": 0.7}
            return
            
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    jur = row["jurisdiction"].strip().lower()
                    try:
                        self.jurisdiction_risk[jur] = float(row["risk_score"])
                    except ValueError:
                        pass
            logger.info(f"Loaded {len(self.jurisdiction_risk)} jurisdictions from {path}")
        except Exception as e:
            logger.error(f"Error loading jurisdiction risk: {e}")
            self.jurisdiction_risk = {"other": 0.7}

    def lookup_jurisdiction_risk(self, jurisdiction: Optional[str]) -> float:
        """Looks up the risk score for a jurisdiction, defaulting to 'Other' (0.7)."""
        if not jurisdiction:
            return self.jurisdiction_risk.get("other", 0.7)
            
        jur_norm = jurisdiction.strip().lower()
        if jur_norm in self.jurisdiction_risk:
            return self.jurisdiction_risk[jur_norm]
            
        # Try substring matching
        for jur, score in self.jurisdiction_risk.items():
            if jur in jur_norm or jur_norm in jur:
                return score
                
        return self.jurisdiction_risk.get("other", 0.7)

    def calculate_one_sidedness_score(self, text: str) -> float:
        """Detects one-sided keywords in text, returning 1.0 if any are present, 0.0 otherwise."""
        text_lower = text.lower()
        keywords = [
            "solely",
            "unilaterally",
            "absolute discretion",
            "at its option",
            "sole option",
            "without liability",
            "sole discretion",
            "unilateral"
        ]
        for kw in keywords:
            if kw in text_lower:
                return 1.0
        return 0.0

    def calculate_market_deviation_score(self, clause_text: str, precedent_passages: List[dict]) -> float:
        """Computes the average cosine distance from the clause to the top precedent passages."""
        if not precedent_passages:
            return 0.0
            
        # Lazy-load sentence transformer model if not provided
        model = self.embedding_model
        if model is None:
            logger.info("Initializing lazy SentenceTransformer model in RiskScorer...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_model = model
            
        clause_emb = model.encode(clause_text)
        
        distances = []
        for p in precedent_passages:
            p_text = p["text"]
            p_emb = model.encode(p_text)
            
            # Cosine Distance = 1.0 - Cosine Similarity
            dot = np.dot(clause_emb, p_emb)
            norm_c = np.linalg.norm(clause_emb)
            norm_p = np.linalg.norm(p_emb)
            
            if norm_c == 0 or norm_p == 0:
                dist = 1.0
            else:
                sim = dot / (norm_c * norm_p)
                dist = float(1.0 - sim)
            distances.append(dist)
            
        return sum(distances) / len(distances) if distances else 0.0

    def calculate_value_risk_score(self, value_usd: Optional[float]) -> float:
        """Calculates normalized value risk: min(value / 500,000, 1.0)."""
        if value_usd is None or value_usd <= 0.0:
            return 0.0
        return min(value_usd / 500000.0, 1.0)

    def score_clause(self, clause: Clause, precedent_passages: List[dict]) -> Dict[str, Any]:
        """Calculates the weighted 4-factor risk score for a single clause."""
        f1 = self.calculate_one_sidedness_score(clause.raw_text)
        f2 = self.calculate_market_deviation_score(clause.raw_text, precedent_passages)
        f3 = self.lookup_jurisdiction_risk(clause.governing_law_jurisdiction)
        
        # Determine if we use 'type' (from classifier risk flags) or 'value' (from contract USD value)
        f_type_or_value = 0.0
        weight_key = "value"
        if "type" in self.weights:
            weight_key = "type"
            if clause.risk_flag == "HIGH_RISK":
                f_type_or_value = 1.0
            elif clause.risk_flag == "REVIEW_REQUIRED":
                f_type_or_value = 0.5
        else:
            f_type_or_value = self.calculate_value_risk_score(clause.contract_value_usd)
            
        score = (
            self.weights.get("one_sidedness", 0.0) * f1 +
            self.weights.get("market_deviation", 0.0) * f2 +
            self.weights.get("jurisdiction", 0.0) * f3 +
            self.weights.get(weight_key, 0.0) * f_type_or_value
        )
        
        return {
            "clause_id": clause.clause_id,
            "raw_text": clause.raw_text,
            "clause_type": clause.clause_type,
            "risk_flag": clause.risk_flag,
            "one_sidedness_score": f1,
            "market_deviation_score": f2,
            "jurisdiction_risk_score": f3,
            "value_risk_score": f_type_or_value if weight_key == "value" else 0.0,
            "type_risk_score": f_type_or_value if weight_key == "type" else 0.0,
            "final_score": float(score)
        }

    def classify_tier(self, score: float) -> str:
        """Classifies a risk score into a Risk Tier."""
        if score < 0.35:
            return "LOW_RISK"
        elif score < 0.60:
            return "REVIEW_REQUIRED"
        else:
            return "HIGH_RISK"

def process_contracts(contracts_dir: str = "contracts", output_csv: str = "contracts/risk_scorecard.csv") -> List[Dict[str, Any]]:
    """Parses, retrieves precedent passages via RAG, scores all contracts, and exports the scorecard."""
    logger.info("Initializing RAG Engine for Risk Scorer...")
    rag = RAGEngine()
    rag.build_or_load_vector_store()
    
    scorer = RiskScorer(rag_engine=rag)
    
    # List all contract files
    all_files = os.listdir(contracts_dir)
    contract_files = [
        f for f in all_files if f.endswith(".pdf") or f.endswith(".docx")
    ]
    
    results = []
    
    for filename in contract_files:
        filepath = os.path.join(contracts_dir, filename)
        contract_id, _ = os.path.splitext(filename)
        logger.info(f"Processing contract: {filename}")
        
        try:
            # 1. Parse and classify clauses
            clauses = parse_contract(filepath)
            if not clauses:
                logger.warning(f"No clauses parsed for contract {contract_id}")
                continue
                
            clause_scores = []
            
            # 2. Score each clause
            for clause in clauses:
                # Retrieve precedents for high risk or review required clauses
                precedents = []
                if clause.risk_flag in ["HIGH_RISK", "REVIEW_REQUIRED"]:
                    # Get top-3 reranked precedent passages
                    precedents = rag.hybrid_search(clause.raw_text, rerank=True, top_n=3)
                
                clause_res = scorer.score_clause(clause, precedents)
                clause_scores.append(clause_res["final_score"])
                
            # 3. Aggregate to contract-level risk score
            contract_score = sum(clause_scores) / len(clause_scores) if clause_scores else 0.0
            tier = scorer.classify_tier(contract_score)
            
            # Use metadata details from the first clause
            first_clause = clauses[0]
            results.append({
                "contract_id": contract_id,
                "contract_type": first_clause.contract_type or "Unknown",
                "counterparty_name": first_clause.counterparty_name or "Unknown",
                "governing_law": first_clause.governing_law_jurisdiction or "Unknown",
                "contract_value_usd": first_clause.contract_value_usd or 0.0,
                "risk_score": round(contract_score, 4),
                "risk_tier": tier
            })
            logger.success(f"Scored {contract_id}: Score={contract_score:.4f}, Tier={tier}")
            
        except Exception as e:
            logger.error(f"Failed to score contract {filename}: {e}")
            
    # Sort results by risk_score descending
    results = sorted(results, key=lambda x: x["risk_score"], reverse=True)
    
    # Export to CSV
    logger.info(f"Exporting scorecard to {output_csv}...")
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "contract_id", "contract_type", "counterparty_name", "governing_law", 
                "contract_value_usd", "risk_score", "risk_tier"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        logger.success(f"Successfully exported {len(results)} rows to {output_csv}")
    except Exception as e:
        logger.error(f"Failed to export scorecard: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Contract Risk Scorer and Scorecard Generator")
    parser.add_argument("--contracts_dir", default="contracts", help="Directory containing PDF/DOCX contracts.")
    parser.add_argument("--output", default="contracts/risk_scorecard.csv", help="Output path for the risk scorecard CSV.")
    args = parser.parse_args()
    
    process_contracts(contracts_dir=args.contracts_dir, output_csv=args.output)

if __name__ == "__main__":
    main()
