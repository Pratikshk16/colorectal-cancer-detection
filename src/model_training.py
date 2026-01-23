import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.logger import get_logger
from src.custom_exception import CustomException

import mlflow # type: ignore
import mlflow.sklearn # type: ignore
class ModelTraining:
    def __init__(self, processed_data_path="artifacts/processed", model_dir="artifacts/model"):
        self.processed_data_path = processed_data_path
        self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = get_logger(__name__)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        self.logger.info("ModelTraining initialized")


    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))

            self.logger.info("Processed data loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading processed data: {e}")
            raise CustomException("Failed to load processed data", e)


    def train_model(self):
        try:
            self.model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs"
            )

            self.model.fit(self.X_train, self.y_train)

            model_path = os.path.join(self.model_dir, "model.pkl")
            joblib.dump(self.model, model_path)

            self.logger.info(f"Model training completed. Saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise CustomException("Failed to train model", e)


    def evaluate_model(self):
        try:
            y_proba = self.model.predict_proba(self.X_test)[:, 1]

            threshold = 0.35
            y_pred = np.where(y_proba >= threshold, "Yes", "No")

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, pos_label="Yes")
            recall = recall_score(self.y_test, y_pred, pos_label="Yes")
            f1 = f1_score(self.y_test, y_pred, pos_label="Yes")
            auc = roc_auc_score(self.y_test, y_proba)

            mlflow.log_metric("Accuracy ", accuracy)
            mlflow.log_metric("Precision ", precision)
            mlflow.log_metric("Recall ", recall)
            mlflow.log_metric("F1 Score ", f1)
            mlflow.log_metric("ROC-AUC ", auc)
            report = classification_report(self.y_test, y_pred)

            self.logger.info("Model Evaluation Metrics:")
            self.logger.info(f"Accuracy : {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall   : {recall:.4f}")
            self.logger.info(f"F1 Score : {f1:.4f}")
            self.logger.info(f"ROC AUC  : {auc:.4f}")
            self.logger.info("Classification Report:\n" + report)

            print("Accuracy:", accuracy)
            print("ROC AUC:", auc)
            print(report)

        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise CustomException("Failed to evaluate model", e)

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

        self.logger.info("Model training pipeline executed successfully")


if __name__ == "__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()
