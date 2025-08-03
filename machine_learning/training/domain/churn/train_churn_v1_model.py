from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from shared.utils.utils import (
    get_X_and_y,
    drop_columns,
    convert_to_numeric_or_coerce_nan,
    convert_binary_yes_no_strings_to_int,
    drop_missing_values,
    clean_column_names,
    create_encoders,
    apply_encoders
)
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

csv_path = Path(__file__).resolve().parents[3] / 'customer_data' / 'churn' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(csv_path)
df = clean_column_names(df)
df = drop_columns(df, ["customerid"])
df = convert_to_numeric_or_coerce_nan(df, ["totalcharges"])
df = convert_binary_yes_no_strings_to_int(df)
df = drop_missing_values(df)

categorical_cols = [
    "gender",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod"
]

encoders = create_encoders(df, categorical_cols)

df = apply_encoders(df, encoders)

encoders_path = Path(__file__).resolve().parents[3] / 'encoders' / 'churn' / 'churn_encoders.pkl'

joblib.dump(encoders, encoders_path)

df = clean_column_names(df)

(X, y) = get_X_and_y(df, "churn")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(classification_report(y_test, y_predict))

model_path = Path(__file__).resolve().parents[3] / 'tabular_models' / 'churn' / 'churn_model_v1.pkl'

joblib.dump(model, model_path)
