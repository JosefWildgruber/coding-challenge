
from sklearn.model_selection import GridSearchCV

from functions import *
from config import *


class TermDepositClassifier:
    def __init__(self, data, pipeline, params=None, expected_columns=None):
        self.data = data
        self.pipeline = pipeline
        self.params = params
        self.expected_columns = expected_columns
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.grid_searches = None
        self.best_estimator = None
        self.prediction_results = None

    def _drop_unknown(self, new_num):
        """Function to drop rows with value 'unknown' in one of the predefined columns.
        Rows are getting dropped only if they make up for less than 5% of the rows in the dataframe.

         :param df: pd.DataFrame, Dataframe with columns to analyse
         :return:
         """
        df_temp = self.data.copy(deep=True)
        for col in new_num:
            df_temp = df_temp[(df_temp[col] == 'unknown')]
            percentage_lost = len(df_temp) / len(self.data)
            print(f'Percentage lost = {percentage_lost}')
        if percentage_lost < 0.05:
            for col in new_num:
                self.data = self.data[(self.data[col] != 'unknown')]

    def _make_numeric(self):
        """Function to make target values numeric

         :param df: pd.DataFrame, Dataframe with columns to analyse
         :return:
         """
        self.data['y'].mask(self.data['y'] == 'yes', 1, inplace=True)
        self.data['y'].mask(self.data['y'] == 'no', 0, inplace=True)

    def _balance_dataset(self, mode='smote'):
        """Function to balance a given dataset in case one target variable is underrepresented.

         :param mode: str, String to select with mode should be used for oversampling of minority class
         :return:
         """
        if mode == 'smote':
            self.X_train, self.y_train = balance_with_smote(self.X_train, self.y_train)
        elif mode == 'duplicates':
            df = pd.concat([self.X_train, self.y_train], axis=1)
            df = balance_with_duplicates(df)
            self.X_train = df.drop('y', axis=1)
            self.y_train = df[['y']]
        else:
            self.X_train = self.X_train,
            self.y_train = self.y_train

    def _balance_with_duplicates(self, df):
        """Function to balance a given dataset by duplicating minority class 10 times.

         :param mode: str, String to select with mode should be used for oversampling of minority class
         :return:
         """
        new_yes = df[df['y'] == 1]
        for i in range(10):
            df = pd.concat([df, new_yes], axis=0)
        return df

    def _balance_with_smote(self, X, y):
        """Function to balance a given dataset by applying SMOTE oversampling.

        :param X: pd.DataFrame, dataframe with train data
        :param y: pd.DataFrame, dataframe with train targets
        :return:
        """
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    def _get_dummies(self):
        """Function to get one-hot-encodings of categorical features of training data.

        :return:
        """
        numerical_features = self.X_train[numerical_columns ]
        categorical_features = self.X_train.drop(numerical_columns, axis=1)
        new_numerical_features = pd.get_dummies(categorical_features)

        self.X_train = pd.concat([numerical_features, new_numerical_features], axis=1)
        self.X_train = self.X_train.reindex(sorted(self.X_train.columns), axis=1)

    def _get_dummies_prediction(self):
        """Function to get one-hot-encodings of categorical features of test data.

        :return:
        """
        numerical_features = self.X_test[numerical_columns]
        categorical_features = self.X_test.drop(numerical_columns, axis=1)
        new_numerical_features = pd.get_dummies(categorical_features)

        self.X_test = pd.concat([numerical_features, new_numerical_features], axis=1)
        self.X_test = self.X_test.reindex(sorted(self.X_test.columns), axis=1)

    def _add_features(self):
        """Function to check if all training features also exist in the test dataframe.
        One-hot encoding might create different columns in both cases. If columns are not present in the test data
        they are getting created and filled with zeros.

        :return:
        """
        for col in self.expected_columns:
            if col not in self.X_test.columns:
                self.X_test[col] = 0
        self.X_test = self.X_test.reindex(sorted(self.X_test.columns), axis=1)

    def _add_lables(self):
        self.prediction_results = pd.DataFrame({'prediction': list(self.prediction_results)})
        self.prediction_results['prediction'].mask(self.prediction_results['prediction'] == 1, 'yes', inplace=True)
        self.prediction_results['prediction'].mask(self.prediction_results['prediction'] == 0, 'no', inplace=True)

    def fit(self, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        self._drop_unknown(columns_to_clean)
        self._make_numeric()
        self.y_train = self.data[['y']].astype(int).values
        self.X_train = self.data.drop(['y'], axis=1)
        self._get_dummies()
        pd.DataFrame({'Columns': self.X_train.columns}).to_csv(f'{model_path}expected_columns.csv')
        self._balance_dataset()
        print("Running GridSearchCV")
        gs = GridSearchCV(self.pipeline, self.params, cv=cv, n_jobs=n_jobs,
                          verbose=verbose, scoring=scoring, refit=refit,
                          return_train_score=True)
        gs.fit(self.X_train, self.y_train)
        print('GS done')
        self.best_estimator = gs.best_estimator_
        self.grid_searches = gs

    def predict(self):
        self._drop_unknown(columns_to_clean)
        self.X_test = self.data
        self._get_dummies_prediction()
        self._add_features()
        self.prediction_results = self.pipeline.predict(self.X_test)
        self._add_lables()
        self.prediction_results = pd.concat([self.X_test, self.prediction_results], axis=1)

    def get_score(self):
        d = {'estimator': None,
             'mean_test_accuracy':None,
             'mean_test_recall':None,
             'mean_test_f1_score':None}

        index = self.grid_searches.cv_results_['params'].index(self.grid_searches.best_params_)
        d['estimator'] = [str(self.grid_searches.best_estimator_._final_estimator)]
        d['mean_test_f1_score'] = [self.grid_searches.cv_results_['mean_test_f1'][index]]
        d['mean_test_accuracy'] = [self.grid_searches.cv_results_['mean_test_accuracy'][index]]
        d['mean_test_recall'] = [self.grid_searches.cv_results_['mean_test_recall'][index]]
        return d









