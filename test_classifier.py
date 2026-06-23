import pytest
import os
from clause_classifier import ClauseClassifier
from config import config

@pytest.fixture(scope="module")
def classifier():
    clf = ClauseClassifier()
    clf.init_classifier()
    return clf

def test_risk_flag_assignment(classifier):
    # Case 1: High risk types with high confidence
    assert classifier.get_risk_flag("Liability", 0.85) == "HIGH_RISK"
    assert classifier.get_risk_flag("Indemnification", 0.61) == "HIGH_RISK"
    assert classifier.get_risk_flag("IP", 0.99) == "HIGH_RISK"
    
    # Case 2: High risk types with low confidence -> Review required
    assert classifier.get_risk_flag("Liability", 0.50) == "REVIEW_REQUIRED"
    assert classifier.get_risk_flag("IP", 0.60) == "REVIEW_REQUIRED"
    
    # Case 3: Non-high risk types with high confidence -> Low risk
    assert classifier.get_risk_flag("Confidentiality", 0.70) == "LOW_RISK"
    assert classifier.get_risk_flag("Governing_Law", 0.60) == "LOW_RISK"
    assert classifier.get_risk_flag("Other", 0.80) == "LOW_RISK"
    
    # Case 4: Non-high risk types with low confidence -> Review required
    assert classifier.get_risk_flag("Confidentiality", 0.45) == "REVIEW_REQUIRED"
    assert classifier.get_risk_flag("Other", 0.59) == "REVIEW_REQUIRED"

def test_prediction_output(classifier):
    # A standard governing law clause should be predicted as Governing_Law
    clause_text = "This contract shall be governed by the laws of the State of Delaware."
    pred_class, confidence, was_fallback = classifier.predict_clause(clause_text)
    
    assert isinstance(pred_class, str)
    assert isinstance(confidence, float)
    assert isinstance(was_fallback, bool)
    assert pred_class == "Governing_Law"

def test_fallback_trigger(classifier):
    # A highly ambiguous or meaningless sentence should trigger the fallback
    ambiguous_text = "Something completely unrelated about standard things and random stuff."
    
    pred_class, confidence, was_fallback = classifier.predict_clause(ambiguous_text)
    
    # Check that fallback was triggered
    assert was_fallback is True
    # The zero-shot NLI classifier should assign some classification
    assert pred_class in [
        "Liability", "Indemnification", "Payment", "Termination", "IP", 
        "Confidentiality", "Governing_Law", "Force_Majeure", "Dispute_Resolution", "Other"
    ]
    assert 0.0 <= confidence <= 1.0
