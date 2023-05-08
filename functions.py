from config import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def feature_importance(X_train, y_train):
    """Function to get each feature's importance for the classification.
    A RandomForestClassifier is used to get the importance of features in the training-set

    :param X_train: pd.DataFrame, Dataframe with features used for classification
    :param y_train: pd.DataFrame, Dataframe with classification targets
    :return:
    """

    feature_names = X_train.columns
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    return forest_importances, std


def preprocessing(data, numerical_columns, new_numerical_columns, columns_to_clean, train=True):

    data_drop = drop_unknown(data, columns_to_clean)
    data_clean = make_numeric(data_drop, new_numerical_columns, train=train)

    if train:
        y_train = data_clean[['y']].astype(int).values
        data_clean.drop(['y'], axis=1, inplace=True)

    numerical_features = data_clean[numerical_columns + new_numerical_columns]
    categorical_features = data_clean.drop(numerical_columns + new_numerical_columns, axis=1)
    new_numerical_features = pd.get_dummies(categorical_features)

    X_res = pd.concat([numerical_features, new_numerical_features], axis=1)
    X_res = X_res.reindex(sorted(X_res.columns), axis=1)

    if train:
        pd.DataFrame({'Columns':X_res.columns}).to_csv(f'{model_path}expected_columns.csv')
        return X_res, y_train

    else:
        return X_res


def generate_column_plots(df):
    """Function to plot value counts for all columns of a Dataframe.
    Plots get stored in graphics folder

     :param df: pd.DataFrame, Dataframe with columns to analyse
     :return:
     """
    for col in df.columns:
        categories = [k for k, v in df[col].value_counts().items()]
        values = [v for k, v in df[col].value_counts().items()]
        fig = plt.figure(figsize=(10, 5))
        # creating the bar plot
        plt.bar(categories, values, width=0.4)
        plt.savefig(f'graphics\\{col}_clean.png')


def generate_correclation_matrix(df):
    """Function to generate a correlation-matrix for a given dataframe.
     The resulting matrix gets stored as .png in the graphics folder.

     :param df: pd.DataFrame, Dataframe with columns to analyse
     :return:
     """
    pear_corr = df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(len(df.columns), len(df.columns)))
    im = ax.imshow(pear_corr, interpolation='nearest')
    fig.colorbar(im, orientation='vertical', fraction=0.05)

    # Show all ticks and label them with the dataframe column name
    ax.set_xticks(range(0, len(df.columns)))
    ax.set_yticks(range(0, len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=65, fontsize=10)
    ax.set_yticklabels(df.columns, rotation=0, fontsize=10)

    # Loop over data dimensions and create text annotations
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            text = ax.text(j, i, round(pear_corr.to_numpy()[i, j], 2),
                           ha="center", va="center", color="black")

    plt.savefig('graphics\\correlation_2.png')


def drop_unknown(df, new_num):
    df_temp = df.copy(deep=True)
    for col in new_num:
        df_temp = df_temp[(df_temp[col] == 'unknown')]
        percentage_lost = len(df_temp)/len(df)
        print(f'Percentage lost = {percentage_lost}')
    if percentage_lost < 0.05:
        for col in new_num:
            df = df[(df[col] != 'unknown')]
        return df
    else:
        return df


def make_numeric(df, new_num, train=True):
    if train:
        df['y'].mask(df['y'] == 'yes', 1, inplace=True)
        df['y'].mask(df['y'] == 'no', 0, inplace=True)
    for col in new_num:
        df[col].mask(df[col] == 'yes', 1, inplace=True)
        df[col].mask(df[col] == 'no', 0, inplace=True)
    return df


def balance_dataset(X, y, mode='smote'):
    if mode == 'smote':
        X_res, y_res = balance_with_smote(X,y)
    elif mode == 'duplicates':
        df = pd.concat([X, y], axis=1)
        df = balance_with_duplicates(df)
        X_res = df.drop('y', axis=1)
        y_res = df[['y']]
    else:
        X_res = X
        y_res = y
    return X_res, y_res


def balance_with_duplicates(df):
    new_yes = df[df['y'] == 1]
    for i in range(10):
        df = pd.concat([df, new_yes], axis=0)
    return df


def balance_with_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def add_features(df, col_list):
    for col in col_list:
        if col not in df.columns:
            df[col]=0
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def add_lables(results_pred):
    results_df = pd.DataFrame({'prediction': list(results_pred)})
    results_df['prediction'].mask(results_df['prediction'] == 1, 'yes', inplace=True)
    results_df['prediction'].mask(results_df['prediction'] == 0, 'no', inplace=True)
    return results_df

