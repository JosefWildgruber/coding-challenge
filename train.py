from model import *
from functions import *
from config import *
import joblib


data = pd.read_excel(train_path)

cls = TermDepositClassifier(data=data,
                            pipeline=train_pipeline,
                            numerical_columns=numerical_columns,
                            params=parameters
                            )

cls.fit(scoring=['accuracy', 'recall', 'f1'],
        refit='recall'
        )

score = cls.get_score()

score.to_csv(f'{model_path}scores.csv')

joblib.dump(cls.best_estimator,
            f'{model_path}pipeline.pkl',
            compress=1
            )

