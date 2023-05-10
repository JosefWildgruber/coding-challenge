from model import *
from config import *
import joblib
import pandas as pd

# Read test data
pred_file = pd.read_excel(test_path)

# Read expected columns
exp_cols = pd.read_csv(f"{model_path}expected_columns.csv")["Columns"]

# Load pipeline
loaded_pipeline = joblib.load(f"{model_path}pipeline.pkl")

# Create classifier object
clf = TermDepositClassifier(
    data=pred_file,
    pipeline=loaded_pipeline,
    numerical_columns=numerical_columns,
    expected_columns=exp_cols,
)

# Run predict
clf.predict()

# Save prediction results
clf.prediction_results.to_csv(f"{output_path}predictions.csv")
