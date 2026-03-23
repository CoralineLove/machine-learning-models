import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def load_data(file_path):
    """Loads data from a CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def split_data(data, test_size=0.2, random_state=42):
    """Splits data into training and testing sets"""
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the performance of a model"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, report, confusion

def run_pipeline(data, pipeline, test_size=0.2, random_state=42):
    """Runs a pipeline on the data"""
    X_train, X_test, y_train, y_test = split_data(data, test_size=test_size, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy, report, confusion = evaluate_model(pipeline, X_test, y_test)
    return accuracy, report, confusion, y_pred

def visualize_data(data):
    """Plots the first two principal components of the data"""
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data.drop("target", axis=1))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data["target"])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization")
    plt.show()