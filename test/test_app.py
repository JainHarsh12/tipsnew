import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Assuming app.py contains the code to train the model.
# We can import specific functions or objects from app if needed.

@pytest.fixture
def sample_data():
    # Load or generate sample data for testing
    file_path = 'tips.csv'
    data = pd.read_csv(file_path)
    return data

def test_data_loading(sample_data):
    # Test if data loads correctly
    assert not sample_data.empty, "The dataset should not be empty"
    assert 'smoker' in sample_data.columns, "Target column 'smoker' should be present in data"

def test_train_test_split(sample_data):
    # Split the data
    X = sample_data.drop(columns='smoker')
    y = sample_data['smoker']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check if split sizes are correct, accounting for rounding
    expected_train_size = int(len(sample_data) * 0.8)
    expected_test_size = len(sample_data) - expected_train_size
    
    assert len(X_train) == expected_train_size, f"Training data size mismatch: expected {expected_train_size}, got {len(X_train)}"
    assert len(X_test) == expected_test_size, f"Test data size mismatch: expected {expected_test_size}, got {len(X_test)}"


def test_pipeline(sample_data):
    # Preprocessing steps and pipeline setup
    X = sample_data.drop(columns='smoker')
    y = sample_data['smoker']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = DecisionTreeClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Assertions to check if pipeline works and accuracy is reasonable
    assert accuracy > 0.6, f"Accuracy should be reasonable, but got {accuracy:.2f}"

    # Check if the pipeline is properly trained
    assert pipeline.named_steps['model'], "Model should be a part of the pipeline"