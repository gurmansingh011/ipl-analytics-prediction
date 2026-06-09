"""Train and save IPL match outcome prediction models."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import IPLDataPreprocessor
from src.feature_engineering import MatchFeatureEngineer


@dataclass
class MatchPredictorTrainer:
    data_dir: Path | str = Path("data")
    model_path: Path | str = Path("models/match_predictor.pkl")
    max_training_rows: int = 80000

    categorical_features = ["batting_team", "bowling_team", "venue"]
    numeric_features = [
        "target_score",
        "current_score",
        "wickets_lost",
        "overs_completed",
        "runs_required",
        "balls_remaining",
        "run_rate",
        "required_run_rate",
    ]

    def train(self) -> dict:
        preprocessor = IPLDataPreprocessor(self.data_dir)
        matches, deliveries, demo_mode = preprocessor.load_data()
        training_df = MatchFeatureEngineer(matches, deliveries).build_training_data()
        if training_df.empty or training_df["result"].nunique() < 2:
            raise ValueError("Not enough completed second-innings data to train the prediction model.")
        if len(training_df) > self.max_training_rows:
            per_class = max(self.max_training_rows // training_df["result"].nunique(), 1)
            sampled_frames = [
                frame.sample(min(len(frame), per_class), random_state=42)
                for _, frame in training_df.groupby("result")
            ]
            training_df = pd.concat(sampled_frames, ignore_index=True).sample(frac=1, random_state=42)

        x = training_df[self.categorical_features + self.numeric_features]
        y = training_df["result"]
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
                max_depth=14,
            ),
        }

        results = {}
        best_name = None
        best_accuracy = -1.0
        best_pipeline = None
        for name, estimator in models.items():
            pipeline = self._pipeline(estimator)
            pipeline.fit(x_train, y_train)
            pred = pipeline.predict(x_test)
            accuracy = accuracy_score(y_test, pred)
            results[name] = {
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            }
            if accuracy > best_accuracy:
                best_name = name
                best_accuracy = accuracy
                best_pipeline = pipeline

        artifact = {
            "model": best_pipeline,
            "best_model_name": best_name,
            "accuracy": best_accuracy,
            "results": results,
            "features": self.categorical_features + self.numeric_features,
            "teams": sorted(pd.concat([matches["team1"], matches["team2"]]).unique()),
            "venues": sorted(matches["venue"].unique()),
            "demo_mode": demo_mode,
        }
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as handle:
            pickle.dump(artifact, handle)
        return artifact

    def _pipeline(self, estimator) -> Pipeline:
        encoder = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
                ("numeric", StandardScaler(), self.numeric_features),
            ]
        )
        return Pipeline([("preprocess", encoder), ("model", estimator)])


if __name__ == "__main__":
    artifact = MatchPredictorTrainer().train()
    print(f"Saved {artifact['best_model_name']} model with accuracy {artifact['accuracy']:.3f}")
