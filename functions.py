from config import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    # fig.tight_layout()
    plt.savefig("graphics\\importance.png")

    return forest_importances, std


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
        plt.savefig(f"graphics\\{col}.png")


def generate_correclation_matrix(df):
    """Function to generate a correlation-matrix for a given dataframe.
    The resulting matrix gets stored as .png in the graphics folder.

    :param df: pd.DataFrame, Dataframe with columns to analyse
    :return:
    """
    pear_corr = df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(len(df.columns), len(df.columns)))
    im = ax.imshow(pear_corr, interpolation="nearest")
    fig.colorbar(im, orientation="vertical", fraction=0.05)

    # Show all ticks and label them with the dataframe column name
    ax.set_xticks(range(0, len(df.columns)))
    ax.set_yticks(range(0, len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=65, fontsize=10)
    ax.set_yticklabels(df.columns, rotation=0, fontsize=10)

    # Loop over data dimensions and create text annotations
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            text = ax.text(
                j,
                i,
                round(pear_corr.to_numpy()[i, j], 2),
                ha="center",
                va="center",
                color="black",
            )

    plt.savefig("graphics\\correlation.png")


def make_numeric(df):
    df["y"].mask(df["y"] == "yes", 1, inplace=True)
    df["y"].mask(df["y"] == "no", 0, inplace=True)
    return df
