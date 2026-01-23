import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.logger import get_logger
from src.custom_exception import CustomException


class DataProcessing:
    """
    Data preprocessing pipeline aligned with the notebook version:
    - Drops Patient_ID
    - Uses a reduced feature set (X_small)
    - OneHotEncodes categorical features
    - Scales numerical features
    - Splits into train/test
    - Saves processed arrays + preprocessor
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.logger = get_logger(__name__)


        self.df = None
        self.X = None
        self.y = None
        self.preprocessor = None

        os.makedirs(self.output_path, exist_ok=True)
        self.logger.info(
            f"DataProcessing initialized | input_path={input_path}, output_path={output_path}"
        )

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load data", e)

    def preprocess_data(self):
        try:

            self.df = self.df.drop(columns=["Patient_ID"], errors="ignore")


            self.y = self.df["Survival_Prediction"]

            important_cols = [
                "Age",
                "Tumor_Size_mm",
                "Cancer_Stage",
                "Treatment_Type",
                "Survival_5_years",
                "Early_Detection",
                "Screening_History",
                "Mortality_Rate_per_100K",
                "Healthcare_Costs",
            ]

            self.X = self.df[important_cols]

            categorical_cols = self.X.select_dtypes(include="object").columns.tolist()
            numerical_cols = self.X.select_dtypes(exclude="object").columns.tolist()

            self.logger.info(f"Categorical columns: {categorical_cols}")
            self.logger.info(f"Numerical columns: {numerical_cols}")

            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_cols,
                    ),
                    ("num", StandardScaler(), numerical_cols),
                ]
            )

            self.logger.info("Preprocessing configuration created successfully")

        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise CustomException("Failed to preprocess data", e)


    def split_and_transform(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y,
            )

            X_train_t = self.preprocessor.fit_transform(X_train)
            X_test_t = self.preprocessor.transform(X_test)

            self.logger.info(
                f"Data split completed | Train: {X_train_t.shape}, Test: {X_test_t.shape}"
            )

            return X_train_t, X_test_t, y_train.values, y_test.values

        except Exception as e:
            self.logger.error(f"Error during splitting/scaling: {e}")
            raise CustomException("Failed to split and transform data", e)

    def save_artifacts(self, X_train, X_test, y_train, y_test):
        try:
            joblib.dump(X_train, os.path.join(self.output_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path, "y_test.pkl"))

            joblib.dump(self.preprocessor, os.path.join(self.output_path, "preprocessor.pkl"))

            self.logger.info("Processed data and preprocessor saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            raise CustomException("Failed to save processed data", e)

    def run(self):
        self.load_data()
        self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_and_transform()
        self.save_artifacts(X_train, X_test, y_train, y_test)

        self.logger.info("Data preprocessing pipeline executed successfully")


if __name__ == "__main__":
    input_path = "artifacts/raw/colorectal_cancer_dataset.csv"
    output_path = "artifacts/processed"

    processor = DataProcessing(input_path, output_path)
    processor.run()
