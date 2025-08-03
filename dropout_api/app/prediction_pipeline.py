import pandas as pd
from shared.utils.utils import ( 
    apply_encoders,
    clean_column_names,
    convert_cols_to_categories,
)

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

def preprocess_data_and_predict(model, encoders, input_data):
    df = preprocess_data(input_data, encoders)
    prediction = make_prediction(model, df)
    return prediction
    
def preprocess_data(input_data, encoders):
    df = pd.DataFrame([input_data])
    df = convert_cols_to_categories(df, categorical_int_cols)
    df = apply_encoders(df, encoders)
    df = clean_column_names(df)
    return df

def make_prediction(model, df):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "dropout_prediction": bool(prediction),
        "dropout_probability": round(probability, 4)
    }