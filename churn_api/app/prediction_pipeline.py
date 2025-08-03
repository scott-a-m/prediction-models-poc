import pandas as pd
from shared.utils.utils import ( 
    apply_encoders,
    convert_binary_yes_no_strings_to_int,
    clean_column_names,
    convert_to_numeric_or_coerce_nan
)

def preprocess_data_and_predict(model, encoders, input_data):
    df = preprocess_data(input_data, encoders)
    prediction = make_prediction(model, df)
    return prediction
    
def preprocess_data(input_data, encoders):
    df = pd.DataFrame([input_data])
    df = convert_to_numeric_or_coerce_nan(df, ["totalcharges"])
    df = apply_encoders(df, encoders)
    df = convert_binary_yes_no_strings_to_int(df)
    df = clean_column_names(df)
    return df

def make_prediction(model, df):
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(probability, 4)
    }