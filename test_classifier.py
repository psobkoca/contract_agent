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
    assert 0.0 <= confidence <= 1.0


def test_train_and_persist(tmp_path):
    # Create a dummy training CSV
    csv_path = tmp_path / "dummy_training.csv"
    model_dir = tmp_path / "models"
    
    # We need 10 samples for each of the 10 classes in self.LABEL_MAP
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        
        classes = [
            "Confidentiality", "Governing Law", "Termination", "Indemnification",
            "Limitation of Liability", "Intellectual Property", "Payment Terms",
            "Force Majeure", "Dispute Resolution", "Assignment"
        ]
        
        for cls in classes:
            for i in range(10):
                writer.writerow([f"This is sample {i} of a standard clause for {cls} and it covers standard things.", cls])
            
    clf = ClauseClassifier(model_dir=str(model_dir), training_csv=str(csv_path))
    # Test initialization with force retraining on our dummy CSV
    clf.init_classifier(force_retrain=True)
    
    # Check that model files were created
    assert os.path.exists(clf.model_path)
    assert os.path.exists(clf.meta_path)
    
    # Check predicting with the newly trained model
    pred, conf = clf.predict_sklearn("This is a Confidentiality agreement.")
    assert pred in clf.classes_


def test_classifier_ac2():
    import csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from config import config
    
    texts, labels = [], []
    with open("clauses_training.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
                
    seed = config.reproducibility.classifier_seed
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]
    
    assert macro_f1 > 0.75


def test_classifier_ac3(classifier):
    fixtures = [
        "Something completely unrelated about standard things and random stuff.",
        "Abcd efgh ijkl mnop qrst uvw xyz.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    ]
    for text in fixtures:
        _, _, fallback_used = classifier.predict_clause(text)
        assert fallback_used is True


