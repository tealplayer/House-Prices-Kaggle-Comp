from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score

import pandas as pd

data = "/Users/srgtchuckles/Downloads/train.csv"
iowa_home_data = pd.read_csv(data)

y = iowa_home_data.SalePrice
X = iowa_home_data.drop(["SalePrice"], axis=1)

qual_scale = ["NA", "Po", "Fa", "TA", "Gd", "Ex"]

ordinal_cols = [
    "ExterQual", "ExterCond",
    "BsmtQual", "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual", "GarageCond",
]
ordinal_orderings = [qual_scale] * len(ordinal_cols)

remaining_cat = [
    c for c in X.columns
    if X[c].dtype == "object" and c not in ordinal_cols
]
low_card_cols = [c for c in remaining_cat if X[c].nunique() < 10]
high_card_cols = [c for c in remaining_cat if X[c].nunique() >= 10]

numerical_cols = [
    c for c in X.columns
    if X[c].dtype in ["int64", "float64"]
]

my_cols = ordinal_cols + low_card_cols + high_card_cols + numerical_cols
X = X[my_cols].copy()

total = len(ordinal_cols) + len(low_card_cols) + len(high_card_cols) + len(numerical_cols)

print(f"ordinal={len(ordinal_cols)}, low={len(low_card_cols)}, "
      f"high={len(high_card_cols)}, num={len(numerical_cols)}, "
      f"sum={total}, my_cols={len(my_cols)}")

assert total == len(my_cols), "Column groups overlap or miss columns"

ordinal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
    ("ordinal", OrdinalEncoder(
        categories=ordinal_orderings,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

low_card_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

high_card_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )),
])

numerical_transformer = SimpleImputer(strategy="median")

preprocessor = ColumnTransformer(transformers=[
    ("ord", ordinal_transformer, ordinal_cols),
    ("low", low_card_transformer, low_card_cols),
    ("high", high_card_transformer, high_card_cols),
    ("num", numerical_transformer, numerical_cols),
])


def get_score(n_estimators):
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        random_state=0,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    scores = -1 * cross_val_score(
        pipeline, X, y,
        cv=5,
        scoring="neg_root_mean_squared_error",
        error_score="raise",
    )

    return scores.mean()


candidate_estimators = [100, 200, 300, 400, 500, 700, 1000]
results = {n: get_score(n) for n in candidate_estimators}

for n, rmse in results.items():
    print(f"n_estimators={n}: RMSE={rmse:.2f}")

best_n = min(results, key=results.get)

print(f"\nBest n_estimators: {best_n} (RMSE={results[best_n]:.2f})")

final = XGBRegressor(n_estimators=best_n, learning_rate=0.05, random_state=0)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', final)])

final_pipeline.fit(X, y)

test_data = pd.read_csv("/Users/srgtchuckles/Downloads/test.csv")

X_test = test_data[my_cols].copy()
test_preds = final_pipeline.predict(X_test)

output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})

output.to_csv("submission.csv", index=False)

print(output.head())
