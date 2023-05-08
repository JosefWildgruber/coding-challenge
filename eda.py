from functions import *
from config import *

data = pd.read_excel(f'{input_path}train_file.xlsx')

data_drop = drop_unknown(data, columns_to_clean)
data_clean = make_numeric(data_drop, new_numerical_columns)

X_raw = data_clean.drop(['y'], axis=1)

numerical_features = X_raw[numerical_columns + new_numerical_columns]
categorical_features = X_raw.drop(numerical_columns + new_numerical_columns, axis=1)

generate_column_plots(data_clean)

new_numerical_features = pd.get_dummies(categorical_features)

X_train = pd.concat([numerical_features, new_numerical_features], axis=1)
y_train = data_clean[['y']].astype(int)

print(X_train.shape)
print(y_train.shape)

generate_correclation_matrix(pd.concat([X_train, y_train], axis=1))

forest_importances, std = feature_importance(X_train=X_train, y_train=y_train)

