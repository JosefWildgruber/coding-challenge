from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Set path
train_path = 'input\\train_file.xlsx'
test_path = 'input\\test_file.xlsx'
model_path = 'trained_models\\'
output_path = 'output\\'

# List numerical columns
numerical_columns = ['age',
                     'duration',
                     'campaign',
                     'previous'
                     ]

# List columns to be cleaned
columns_to_clean = ['loan',
                    'housing',
                    'marital',
                    'job',
                    'education',
                    'default'
                    ]

# Set balancing strategy
balancing = 'smote'

# Set pipeline for grid-search
train_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', LogisticRegression()),
])

# Set parameters for grid-search
parameters = [
    {
        'clf': (LogisticRegression(max_iter=10000,
                                   tol=0.1),
                ),
        'clf__C': (0.01,0.1,1,10),
        'pca__n_components': [5, 15],
    }, {
        'clf': (RandomForestClassifier(),
                ),
        'clf__n_estimators': (16, 32),
        'pca__n_components': [5, 15],
    }, {
        'clf': (GradientBoostingClassifier(),
                ),
        'clf__n_estimators': (16, 32),
        'clf__learning_rate': (0.8, 1.0),
        'pca__n_components': [5, 15],
    }, {
        'clf': (MLPClassifier(solver='lbfgs',
                              learning_rate='adaptive',
                              random_state=0),
                ),
        'clf__hidden_layer_sizes': ((32, 2),
                                    (64, 32, 2)
                                    ),
        'pca__n_components': [5, 15],
    }, {
        'clf': (KNeighborsClassifier(3),
                ),
        'pca__n_components': [5, 15],
    }, {
        'clf': (SVC(kernel="linear", C=0.025),
                ),
        'pca__n_components': [5, 15],
    }#, {
#        'clf': (SVC(gamma=2, C=1),
#                ),
#        'pca__n_components': [5, 15],
#    }, {
#        'clf': (GaussianProcessClassifier(1.0 * RBF(1.0)),
#                ),
#        'pca__n_components': [5, 15],
#    }, {
#        'clf': (DecisionTreeClassifier(max_depth=5),
#                ),
#        'pca__n_components': [5, 15],
#    }, {
#        'clf': (GaussianNB(),
#                ),
#        'pca__n_components': [5, 15],
#    }, {
#        'clf': (QuadraticDiscriminantAnalysis(),
#                ),
#        'pca__n_components': [5, 15],
#    }
]

