
from sklearn.model_selection import GridSearchCV

from functions import *
from config import *


class TermDepositClassifier:
    def __init__(self, data: pd.DataFrame, pipeline: Pipeline, numerical_columns: list, params=None, expected_columns=None):
        self.data = data
        self.pipeline = pipeline
        self.params = params
        self.expected_columns = expected_columns
        self.numerical_columns = numerical_columns
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.grid_searches = None
        self.best_estimator = None
        self.prediction_results = None

    def _drop_unknown(self, data: pd.DataFrame, col_to_clean: list) -> pd.DataFrame:
        """Function to drop rows with value 'unknown' in one of the predefined columns.
        Rows are getting dropped only if they their total makes up for less than 5% of the rows in the dataframe.

         :param df: pd.DataFrame, Dataframe with columns with unknowns
         :return: pd.DataFarme: Cleaned dataframe
         """

        df_temp = data.copy(deep=True)
        percentage_lost = 0
        # Loop over columns to  and sum up percentage of unknowns
        for col in col_to_clean:
            df_temp = df_temp[(df_temp[col] == 'unknown')]
            percentage_lost += len(df_temp) / len(self.data)
        print(f'Percentage lost = {percentage_lost}')
        if percentage_lost <= 0.05:
            for col in col_to_clean:
                data = data[(data[col] != 'unknown')]
        return data

    def _make_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        """Function to make target values numeric

         :param df: pd.DataFrame, Dataframe with classification target
         :return: pd.DataFreame, Dataframe with binary target
         """

        # Make target column binary
        data['y'].mask(data['y'] == 'yes', 1, inplace=True)
        data['y'].mask(data['y'] == 'no', 0, inplace=True)

        return data

    def _balance_dataset(self, X: pd.DataFrame, y: pd.DataFrame, mode='smote') -> (pd.DataFrame, pd.DataFrame):
        """Function to balance a given dataset in case one target variable is underrepresented.

         :param mode: str, String to select with mode should be used for oversampling of minority class
         :return: X: pd.DataFrame, Balanced training data for classification
                  y: pd.DataFrame, Balanced targets for classification
         """
        if mode == 'smote':
            X, y = balance_with_smote(X, y)
        elif mode == 'duplicates':
            df = pd.concat([X, y], axis=1)
            df = balance_with_duplicates(df)
            X = df.drop('y', axis=1)
            y = df[['y']]
        else:
            X = X
            y = y

        return X, y

    def _balance_with_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to balance a given dataset by duplicating minority class 10 times.

         :param mode: str, String to select with mode should be used for oversampling of minority class
         :return: df: pd.DataFrame, Balanced data with duplicates
         """
        new_yes = df[df['y'] == 1]
        for i in range(10):
            df = pd.concat([df, new_yes], axis=0)
        return df

    def _balance_with_smote(self, X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """Function to balance a given dataset by applying SMOTE oversampling.

        :param X: pd.DataFrame, dataframe with train data
        :param y: pd.DataFrame, dataframe with train targets
        :return: X_res: pd.DataFramee, dataframe with SMOTE balanced train data
                y_res: pd.DataFramee, dataframe with SMOTE balanced targets
        """
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    def _get_dummies(self, X: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
        """Function to get one-hot-encodings of categorical features of training data.

        :return: X: pd.DataFrame, one hot encoded train data
        """
        numerical_features = X[numerical_columns]
        categorical_features = X.drop(numerical_columns, axis=1)
        new_numerical_features = pd.get_dummies(categorical_features)

        X = pd.concat([numerical_features, new_numerical_features], axis=1)
        X = X.reindex(sorted(X.columns), axis=1)

        return X

    def _add_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Function to check if all training features also exist in the test dataframe.
        One-hot encoding might create different columns in both cases. If columns are not present in the test data
        they are getting created and filled with zeros.

        :return: X: pd.DataFrame, Test dataset with same features as train dataset
        """
        for col in self.expected_columns:
            if col not in X.columns:
                X[col] = 0
        X = X.reindex(sorted(X.columns), axis=1)
        return X

    def _add_labels(self, results) -> pd.DataFrame:
        """Change target predictions from binary to 'yes' and 'no'.

        :return: X: pd.DataFrame, Predicted targets with labels 'yes' and 'no'
        """
        results = pd.DataFrame({'prediction': list(results)})
        results['prediction'].mask(results['prediction'] == 1, 'yes', inplace=True)
        results['prediction'].mask(results['prediction'] == 0, 'no', inplace=True)
        return results

    def fit(self, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        self.data = self._drop_unknown(data=self.data, col_to_clean=columns_to_clean)
        print('Dropping unknowns: Done')
        self.data = self._make_numeric(data=self.data)
        print('Make binary target: Done')
        self.y_train = self.data[['y']].astype(int).values
        self.X_train = self.data.drop(['y'], axis=1)
        print('Created x and y for training: Done')
        self.X_train = self._get_dummies(X=self.X_train, numerical_columns=self.numerical_columns)
        print('Create One-Hot Encodings: Done')
        pd.DataFrame({'Columns': self.X_train.columns}).to_csv(f'{model_path}expected_columns.csv')
        print('Save list of expected columns: Done')
        self.X_train, self.y_train = self._balance_dataset(X=self.X_train, y=self.y_train, mode='smote')
        print("Running GridSearchCV")
        gs = GridSearchCV(self.pipeline, self.params, cv=cv, n_jobs=n_jobs,
                          verbose=verbose, scoring=scoring, refit=refit,
                          return_train_score=True)
        gs.fit(self.X_train, self.y_train)
        print('Running GridSearchCV: Done')
        self.best_estimator = gs.best_estimator_
        self.grid_searches = gs

    def predict(self):
        self.data = self._drop_unknown(data=self.data, col_to_clean=columns_to_clean)
        print('Dropping unknowns: Done')
        self.X_test = self.data
        self.X_test = self._get_dummies(X=self.X_test, numerical_columns=self.numerical_columns)
        print('Create One-Hot Encodings: Done')
        self.X_test = self._add_features(X=self.X_test)
        print('Add missing features: Done')
        self.prediction_results = self.pipeline.predict(self.X_test)
        self.prediction_results = self._add_labels(results=self.prediction_results)
        self.prediction_results = pd.concat([self.X_test, self.prediction_results], axis=1)
        print('Create final results: Done')

    def get_score(self):
        """Function to collect parameters, training recall, training accuracy and training f1 scores for all models.

        :return: scores: pd.DataFrame, Dataframe with training recall, training accuracy
                 and training f1 scores for all models
        """
        d = {'estimator': [],
             'mean_test_accuracy': [],
             'mean_test_recall': [],
             'mean_test_f1_score': []}

        for item in self.grid_searches.cv_results_['params']:
            index = self.grid_searches.cv_results_['params'].index(item)
            estimator = ''
            for para in item:
                estimator = f'{estimator}, {item[para]}'
            d['estimator'].append(estimator)
            d['mean_test_f1_score'].append(self.grid_searches.cv_results_['mean_test_f1'][index])
            d['mean_test_accuracy'].append(self.grid_searches.cv_results_['mean_test_accuracy'][index])
            d['mean_test_recall'].append(self.grid_searches.cv_results_['mean_test_recall'][index])

        scores = pd.DataFrame(d).sort_values(by=['mean_test_recall'])
        return scores









