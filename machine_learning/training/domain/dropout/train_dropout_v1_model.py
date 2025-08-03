from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from shared.utils.utils import (
    get_X_and_y, 
    convert_cols_to_categories, 
    clean_column_names,
    apply_encoders,
    create_encoders
)
import pandas as pd

csv_path = Path(__file__).resolve().parents[3] / 'customer_data' / 'dropout' / 'dropout_data.csv'

df = pd.read_csv(csv_path, sep=";")

df = clean_column_names(df)

print(df.dtypes.to_dict())
categorical_int_cols = [
    "marital_status",
    "application_mode",
    "application_order",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "gender"
]

df = convert_cols_to_categories(df, categorical_int_cols)

df["target"] = df["target"].map({"Dropout": 1, "Graduate": 0, "Enrolled": 0})

encoders = create_encoders(df, categorical_int_cols)

# need to save the encoders for future use

encoders_path = Path(__file__).resolve().parents[3] / 'encoders' / 'dropout' / 'dropout_encoders.pkl'

joblib.dump(encoders, encoders_path)

df = apply_encoders(df, encoders)

df = clean_column_names(df)

print(df["gdp"].head(100))


(X, y) = get_X_and_y(df, "target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))

model_path = Path(__file__).resolve().parents[3] / 'tabular_models' / 'dropout' / 'dropout_model_v1.pkl'

joblib.dump(model, model_path)