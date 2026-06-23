import os
import csv
import json
import pickle
import hashlib
from typing import Tuple, Dict, Any, Optional
import numpy as np
from loguru import logger

class ClauseClassifier:
    """Clause Classifier using TF-IDF + Logistic Regression with fallback support."""
    
    LABEL_MAP = {
        "Confidentiality": "Confidentiality",
        "Governing Law": "Governing_Law",
        "Termination": "Termination",
        "Indemnification": "Indemnification",
        "Limitation of Liability": "Liability",
        "Intellectual Property": "IP",
        "Payment Terms": "Payment",
        "Force Majeure": "Force_Majeure",
        "Dispute Resolution": "Dispute_Resolution",
        "Assignment": "Other"
    }

    def __init__(self, model_dir: str = "./models", training_csv: str = "clauses_training.csv"):
        self.model_dir = model_dir
        self.training_csv = training_csv
        self.model_path = os.path.join(model_dir, "clause_classifier.pkl")
        self.meta_path = os.path.join(model_dir, "classifier_meta.json")
        self.pipeline = None
        self.classes_ = None
        self.nli_pipeline = None

    def get_nli_pipeline(self):
        """Lazy-loads the zero-shot NLI classifier pipeline."""
        if self.nli_pipeline is None:
            from transformers import pipeline
            logger.info("Loading zero-shot classification pipeline (cross-encoder/nli-distilroberta-base)...")
            self.nli_pipeline = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
        return self.nli_pipeline

    def predict_sklearn(self, text: str) -> Tuple[str, float]:
        """Predicts the clause type and confidence using only the trained Scikit-Learn classifier model."""
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been initialized. Call init_classifier() first.")

        probs = self.pipeline.predict_proba([text])[0]
        max_idx = np.argmax(probs)
        pred_class = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        return pred_class, confidence

    def predict_nli_fallback(self, text: str) -> Tuple[str, float]:
        """Predicts the clause type and confidence using the zero-shot NLI fallback classifier."""
        candidate_labels = list(self.classes_) if self.classes_ else [
            "Liability", "Indemnification", "Payment", "Termination", "IP", 
            "Confidentiality", "Governing_Law", "Force_Majeure", "Dispute_Resolution", "Other"
        ]
        
        nli = self.get_nli_pipeline()
        res = nli(text, candidate_labels=candidate_labels)
        
        nli_class = res["labels"][0]
        nli_confidence = float(res["scores"][0])
        return nli_class, nli_confidence

    def predict_clause(self, text: str) -> Tuple[str, float, bool]:
        """Predicts the clause type and confidence for a given text, utilizing NLI fallback if confidence is low."""
        from config import config
        confidence_threshold = config.classifier.confidence_threshold

        pred_class, confidence = self.predict_sklearn(text)

        # Check if confidence meets threshold
        if confidence >= confidence_threshold:
            return pred_class, confidence, False

        # Fallback to Zero-Shot NLI
        logger.warning(
            f"Low confidence ({confidence:.2f} < {confidence_threshold:.2f}) for clause: '{text[:60]}...'. "
            "Triggering Zero-Shot NLI fallback..."
        )
        
        nli_class, nli_confidence = self.predict_nli_fallback(text)
        logger.success(f"NLI Fallback completed: predicted '{nli_class}' with confidence {nli_confidence:.2f}")
        return nli_class, nli_confidence, True

    def get_risk_flag(self, clause_type: str, confidence: float) -> str:
        """Assigns risk flag based on clause type and prediction confidence."""
        from config import config
        high_risk_types = config.classifier.high_risk_types
        
        if clause_type in high_risk_types:
            if confidence > 0.60:
                return "HIGH_RISK"
            else:
                return "REVIEW_REQUIRED"
        else:
            if confidence >= 0.60:
                return "LOW_RISK"
            else:
                return "REVIEW_REQUIRED"

    def get_csv_hash(self) -> str:
        """Computes SHA-256 hash of the training CSV file."""
        if not os.path.exists(self.training_csv):
            raise FileNotFoundError(f"Training CSV not found: {self.training_csv}")
        
        hasher = hashlib.sha256()
        with open(self.training_csv, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def train_and_persist(self) -> None:
        """Trains TF-IDF + Logistic Regression pipeline and persists it along with CSV hash."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score

        logger.info(f"Loading training data from {self.training_csv}...")
        texts, labels = [], []
        with open(self.training_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                raw_label = row.get("label", "").strip()
                if text and raw_label:
                    clean_label = self.LABEL_MAP.get(raw_label, raw_label.replace(" ", "_"))
                    texts.append(text)
                    labels.append(clean_label)

        if not texts:
            raise ValueError(f"No valid training data found in {self.training_csv}")

        from config import config
        seed = config.reproducibility.classifier_seed

        # Compute validation metrics on a train/test split (FR-08)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=seed, stratify=labels
        )
        
        val_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
        ])
        val_pipeline.fit(X_train, y_train)
        y_pred = val_pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logger.info("--- Model Validation Metrics (Train/Test Split) ---")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict):
                logger.info(
                    f"Class '{cls_name}': Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, F1-Score={metrics['f1-score']:.4f}"
                )

        # Train final model on 100% of the data
        logger.info(f"Fitting final Logistic Regression model on all {len(texts)} samples...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=seed))
        ])
        
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.classes_
        
        # Persist model
        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.success(f"Model persisted to {self.model_path}")

        # Compute hash and persist metadata
        csv_hash = self.get_csv_hash()
        meta = {
            "training_csv_hash": csv_hash,
            "classes": list(self.classes_)
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.success(f"Metadata persisted to {self.meta_path}")

    def load_model(self) -> bool:
        """Loads model from disk if it exists and matches training CSV hash."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            logger.info("Model or metadata file not found on disk.")
            return False

        try:
            current_hash = self.get_csv_hash()
        except FileNotFoundError:
            logger.warning(f"Training CSV {self.training_csv} not found. Cannot verify hash.")
            return False

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            saved_hash = meta.get("training_csv_hash")
            if saved_hash != current_hash:
                logger.info("Training CSV hash mismatch. Invalidation triggered.")
                return False
                
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)
            self.classes_ = self.pipeline.classes_
            logger.success("Model successfully loaded from disk cache.")
            return True
        except Exception as e:
            logger.error(f"Error loading model from disk: {e}")
            return False
            
    def init_classifier(self, force_retrain: bool = False) -> None:
        """Initializes the classifier, reloading or retraining as necessary."""
        if force_retrain:
            logger.info("Force retrain requested.")
            self.train_and_persist()
        else:
            if not self.load_model():
                logger.info("Retraining model from scratch...")
                self.train_and_persist()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clause Classifier Trainer and CLI")
    parser.add_argument("--retrain", action="store_true", help="Force retraining the classifier from scratch.")
    args = parser.parse_args()

    # Configure loguru to write to agent.log
    logger.add("agent.log", rotation="10 MB", level="INFO")
    
    classifier = ClauseClassifier()
    classifier.init_classifier(force_retrain=args.retrain)

if __name__ == "__main__":
    main()
