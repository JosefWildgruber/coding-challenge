from model import *
from config import *
import joblib
import pandas as pd

pred_file = pd.read_excel(test_path)
exp_cols = pd.read_csv(f'{model_path}expected_columns.csv')['Columns']
loaded_pipeline = joblib.load(f'{model_path}pipeline.pkl')

clf = TermDepositClassifier(data=pred_file,
                            pipeline=loaded_pipeline,
                            numerical_columns=numerical_columns,
                            expected_columns=exp_cols
                            )

clf.predict()
clf.prediction_results.to_csv(f'{output_path}predictions.csv')
