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

        logger.info(f"Loading training data from {self.training_csv}...")
        texts, labels = [], []
        with open(self.training_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                raw_label = row.get("label", "").strip()
                if text and raw_label in self.LABEL_MAP:
                    texts.append(text)
                    labels.append(self.LABEL_MAP[raw_label])

        if not texts:
            raise ValueError(f"No valid training data found in {self.training_csv}")

        logger.info(f"Training Logistic Regression pipeline on {len(texts)} samples...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, C=1.0))
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
    classifier = ClauseClassifier()
    classifier.init_classifier()

if __name__ == "__main__":
    main()
