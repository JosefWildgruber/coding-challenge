from model import *
from functions import *
from config import *
import joblib

# read training data
data = pd.read_excel(train_path)

# Create classifier object
cls = TermDepositClassifier(data=data,
                            pipeline=train_pipeline,
                            numerical_columns=numerical_columns,
                            params=parameters
                            )

# Run preprocessing steps and grid search
cls.fit(scoring=['accuracy', 'recall', 'f1'],
        refit='recall'
        )

# Get scores
score = cls.get_score()

# Save scores
score.to_csv(f'{model_path}scores.csv')

# Save best pipeline
joblib.dump(cls.best_estimator,
            f'{model_path}pipeline.pkl',
            compress=1
            )

