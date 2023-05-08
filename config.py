from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


train_path = 'input\\train_file.xlsx'
test_path = 'input\\test_file.xlsx'
model_path = 'trained_models\\'
output_path = 'output\\'

numerical_columns = ['age', 'duration', 'campaign', 'previous']
columns_to_clean = ['loan', 'housing', 'marital']
new_numerical_columns = []

balancing = 'smote'

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression()),
])
parameters = [
    {
        'clf': (LogisticRegression(max_iter=10000, tol=0.1),),
        'clf__C': (0.001,0.01,0.1,1,10,100),
        'pca__n_components': [5, 15],
    }, {
        'clf': (RandomForestClassifier(),),
        'clf__n_estimators': (16, 32),
        'pca__n_components': [5, 15],
    }, {
        'clf': (GradientBoostingClassifier(),),
        'clf__n_estimators': (16, 32),
        'clf__learning_rate': (0.8, 1.0),
        'pca__n_components': [5, 15],
    }, {
        'clf': (MLPClassifier(solver='lbfgs', learning_rate='adaptive', random_state=0),),
        'clf__hidden_layer_sizes': ((32,16,2),(64,32,16,2)),
        'pca__n_components': [5, 15],
    }
]
