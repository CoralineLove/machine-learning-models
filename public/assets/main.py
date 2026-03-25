import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def load_data(data_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Successfully loaded data from {data_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preprocesses the data, separating features and target."""
    try:
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
        feature_names = list(df.drop(columns=[target_column]).columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logging.info("Data preprocessing completed.")
        return X_scaled, y, feature_names
    except KeyError:
        logging.error(f"Target column '{target_column}' not found in the DataFrame.")
        raise
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """Trains a Logistic Regression model."""
    try:
        model = LogisticRegression(random_state=42, solver='liblinear', C=C)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        logging.info(f"Model training completed in {training_time:.2f} seconds.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluates the trained model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        logging.info(f"Model accuracy: {accuracy:.4f}")
        logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        return {"accuracy": accuracy, "classification_report": report}
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise


def save_model(model: LogisticRegression, model_path: str):
    """Saves the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        raise


def main(data_path: str, target_column: str, model_path: str, test_size: float, C: float):
    """Main function to load, preprocess, train, evaluate, and save the model."""
    try:
        df = load_data(data_path)
        X, y, _ = preprocess_data(df, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = train_model(X_train, y_train, C)
        evaluate_model(model, X_test, y_test)
        save_model(model, model_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Logistic Regression model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file.")
    parser.add_argument(
        "--target_column", type=str, required=True, help="Name of the target column."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to save the trained model."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2).",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength for Logistic Regression (default: 1.0).",
    )
    args = parser.parse_args()

    main(args.data_path, args.target_column, args.model_path, args.test_size, args.C)