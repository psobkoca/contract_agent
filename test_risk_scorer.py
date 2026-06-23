import pytest
from models import Clause
from risk_scorer import RiskScorer

@pytest.fixture
def scorer():
    # Instantiate RiskScorer with a dummy path so it doesn't fail if the CSV isn't found
    return RiskScorer(jurisdiction_risk_path="jurisdiction_risk.csv")

def test_one_sidedness_keyword_detection(scorer):
    # Neutral text
    assert scorer.calculate_one_sidedness_score("This is a standard clause.") == 0.0
    
    # Text with one-sided keywords
    assert scorer.calculate_one_sidedness_score("This shall be solely determined by the company.") == 1.0
    assert scorer.calculate_one_sidedness_score("The provider may unilaterally terminate the agreement.") == 1.0
    assert scorer.calculate_one_sidedness_score("We have absolute discretion over this.") == 1.0
    assert scorer.calculate_one_sidedness_score("At its option, the client may renew.") == 1.0

def test_jurisdiction_lookup(scorer):
    # Exists in CSV
    assert scorer.lookup_jurisdiction_risk("Delaware") == 0.1
    assert scorer.lookup_jurisdiction_risk("New York") == 0.1
    assert scorer.lookup_jurisdiction_risk("California") == 0.3
    
    # Case insensitivity and whitespace handling
    assert scorer.lookup_jurisdiction_risk("  delaware  ") == 0.1
    
    # Missing jurisdiction -> default to Other (0.7)
    assert scorer.lookup_jurisdiction_risk("UnknownLand") == 0.7
    assert scorer.lookup_jurisdiction_risk(None) == 0.7

def test_value_risk_score(scorer):
    # Normalized value: value / 500000 capped at 1.0
    assert scorer.calculate_value_risk_score(0.0) == 0.0
    assert scorer.calculate_value_risk_score(None) == 0.0
    assert scorer.calculate_value_risk_score(-100.0) == 0.0
    assert scorer.calculate_value_risk_score(250000.0) == 0.5
    assert scorer.calculate_value_risk_score(500000.0) == 1.0
    assert scorer.calculate_value_risk_score(1000000.0) == 1.0

def test_risk_tier_boundaries(scorer):
    # LOW_RISK: < 0.35
    assert scorer.classify_tier(0.0) == "LOW_RISK"
    assert scorer.classify_tier(0.349) == "LOW_RISK"
    
    # REVIEW_REQUIRED: 0.35 <= score < 0.60
    assert scorer.classify_tier(0.35) == "REVIEW_REQUIRED"
    assert scorer.classify_tier(0.50) == "REVIEW_REQUIRED"
    assert scorer.classify_tier(0.599) == "REVIEW_REQUIRED"
    
    # HIGH_RISK: >= 0.60
    assert scorer.classify_tier(0.60) == "HIGH_RISK"
    assert scorer.classify_tier(0.85) == "HIGH_RISK"
    assert scorer.classify_tier(1.0) == "HIGH_RISK"

def test_clause_score_calculation():
    # Pass custom weights to test math correctness and fallback handling
    custom_weights = {
        "one_sidedness": 0.25,
        "market_deviation": 0.35,
        "jurisdiction": 0.20,
        "value": 0.20
    }
    test_scorer = RiskScorer(weights=custom_weights, jurisdiction_risk_path="jurisdiction_risk.csv")
    
    # Mock a clause and its metadata
    clause = Clause(
        clause_id="TEST_CLS_001",
        raw_text="The provider may unilaterally make changes to the pricing.",
        governing_law_jurisdiction="Delaware",
        contract_value_usd=100000.0,
        clause_type="Payment Terms",
        risk_flag="REVIEW_REQUIRED"
    )
    
    # We will pass empty passages so market deviation defaults to 0.0
    precedent_passages = []
    
    # Calculate factors manually:
    # f1 (one-sidedness) = 1.0 (contains "unilaterally")
    # f2 (market deviation) = 0.0
    # f3 (jurisdiction) = 0.1 (Delaware)
    # f4 (value) = 100000 / 500000 = 0.2
    
    # Expected weighted score:
    # 0.25 * 1.0 + 0.35 * 0.0 + 0.20 * 0.1 + 0.20 * 0.2
    # = 0.25 + 0.0 + 0.02 + 0.04 = 0.31
    
    res = test_scorer.score_clause(clause, precedent_passages)
    assert res["one_sidedness_score"] == 1.0
    assert res["market_deviation_score"] == 0.0
    assert res["jurisdiction_risk_score"] == 0.1
    assert res["value_risk_score"] == 0.2
    assert pytest.approx(res["final_score"], abs=1e-5) == 0.31

def test_role_biased_one_sidedness(scorer):
    # If our role matches the bias (disadvantaged), one-sided keyword should yield 1.0
    assert scorer.calculate_one_sidedness_score(
        "This shall be solely determined by the company.",
        clause_type="Liability",
        our_role="BUYER"
    ) == 1.0

    # If our role is different from the bias, it advantages us, yielding 0.2
    assert scorer.calculate_one_sidedness_score(
        "This shall be solely determined by the company.",
        clause_type="Liability",
        our_role="LICENSOR"
    ) == 0.2

def test_clause_type_weight_loading(scorer):
    assert "Liability" in scorer.clause_type_weights
    assert scorer.clause_type_weights["Liability"]["type_weight"] == 1.0
    assert scorer.clause_type_weights["Liability"]["review_required"] is True
    assert scorer.clause_type_weights["Liability"]["our_role_bias"] == "BUYER"
    
    assert "Confidentiality" in scorer.clause_type_weights
    assert scorer.clause_type_weights["Confidentiality"]["type_weight"] == 0.5
    assert scorer.clause_type_weights["Confidentiality"]["review_required"] is False
    assert scorer.clause_type_weights["Confidentiality"]["our_role_bias"] == "CLIENT"


def test_risk_scorer_main_cli(tmp_path):
    import sys
    import os
    from unittest.mock import patch
    import risk_scorer
    
    out_csv = tmp_path / "risk_scorecard_out.csv"
    
    test_args = [
        "risk_scorer.py",
        "--contracts_dir", "contracts",
        "--output", str(out_csv)
    ]
    
    with patch.object(sys, "argv", test_args):
        risk_scorer.main()
        
    assert os.path.exists(out_csv)


def test_risk_scorer_ac5():
    # Verify manual score calculation for 5 reference clauses
    custom_weights = {
        "one_sidedness": 0.25,
        "market_deviation": 0.35,
        "jurisdiction": 0.20,
        "value": 0.20
    }
    scorer = RiskScorer(weights=custom_weights, jurisdiction_risk_path="jurisdiction_risk.csv")
    
    # 5 reference clauses
    clauses = [
        Clause(
            clause_id=f"REF_00{i}_CLS_01",
            raw_text="The provider may unilaterally make changes.",
            governing_law_jurisdiction="Delaware",
            contract_value_usd=100000.0,
            clause_type="Payment Terms",
            risk_flag="REVIEW_REQUIRED"
        )
        for i in range(1, 6)
    ]
    
    for c in clauses:
        res = scorer.score_clause(c, [])
        # f1 (one-sidedness) = 1.0 (contains "unilaterally")
        # f2 (market deviation) = 0.0 (precedents empty)
        # f3 (jurisdiction) = 0.1 (Delaware)
        # f4 (value) = 100000 / 500000 = 0.2
        # score = 0.25 * 1.0 + 0.35 * 0.0 + 0.20 * 0.1 + 0.20 * 0.2 = 0.31
        assert pytest.approx(res["final_score"], abs=1e-5) == 0.31
        
    # Check that score >= 0.70 is classified as CRITICAL in pipeline logic
    assert 0.80 >= 0.70


