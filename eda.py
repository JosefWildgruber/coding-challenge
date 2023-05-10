from functions import *
from config import *

# Read data
data = pd.read_excel(train_path)

# Change target to binary
data_clean = make_numeric(data)

# Save column stats
stats = data_clean[numerical_columns].describe()
stats.to_csv("graphics\\stats.csv")

# Save columns plots
count = data_clean["y"].value_counts()
generate_column_plots(data_clean)

# Create one-hot-encodings
X_raw = data_clean.drop(["y"], axis=1)
numerical_features = X_raw[numerical_columns]
categorical_features = X_raw.drop(numerical_columns, axis=1)
new_numerical_features = pd.get_dummies(categorical_features)

# Create X and y
X_train = pd.concat([numerical_features, new_numerical_features], axis=1)
y_train = data_clean[["y"]].astype(int).values

# Create and save correlation matrix
generate_correclation_matrix(pd.concat([X_train, y_train], axis=1))

# Create and save feature importance plot
forest_importances, std = feature_importance(X_train=X_train, y_train=y_train)
