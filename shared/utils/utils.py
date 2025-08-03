import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data(path, sep=","):
    df = pd.read_csv(path, sep)
    return df

def drop_columns(df, columns):
    return df.drop(columns=columns, axis=1)

def convert_to_numeric_or_coerce_nan(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def convert_binary_yes_no_strings_to_int(df):
    for col in df.columns:
        unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
        if unique_vals <= {"yes", "no"}:
            df[col] = df[col].str.lower().map({"yes": 1, "no": 0})
    return df

def convert_cols_to_categories(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def drop_missing_values(df):
    return df.dropna()

def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
            .str.strip()  # remove leading/trailing whitespace
            .str.replace("'", "", regex=False)  # remove apostrophes
            .str.replace(r"[^\w]", "_", regex=True)  # replace other non-word chars with _
            .str.replace(r"__+", "_", regex=True)  # collapse multiple underscores
            .str.rstrip("_")  # remove trailing underscores
            .str.lower()  # convert to lowercase
    )
    return df

def get_X_and_y(df, targetVector):
    return (df.drop(targetVector, axis=1), df[targetVector])

def check_stratification(y_train, y_test):
    print("Train class distribution:\n", y_train.value_counts(normalize=True))
    print("Test class distribution:\n", y_test.value_counts(normalize=True))

def create_encoders(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        enc.fit(df[[col]])
        encoders[col] = enc
    return encoders

def apply_encoders(df, encoders):
    encoded_parts = []
    for col, enc in encoders.items():
        try:
            encoded = enc.transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=enc.get_feature_names_out([col]),
                index=df.index
            )
            encoded_parts.append(encoded_df)
        except Exception as e:
            print(f"❌ Encoding failed for column '{col}'")
            print(f"   - First value: {df[col].iloc[0]} (type: {type(df[col].iloc[0])})")
            print(f"   - Encoder categories: {enc.categories_[0]}")
            raise ValueError(f"Encoding failed for column '{col}': {e}")
    df_encoded = pd.concat(encoded_parts, axis=1)
    df_numeric = df.drop(columns=encoders.keys())
    return pd.concat([df_encoded, df_numeric], axis=1)

def diagnose_columns(df, title="Column Diagnostics"):
    print(f"\n--- {title} ---")
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=False)
        missing_vals = df[col].isna().sum()
        dtype = df[col].dtype
        constant = unique_vals == 1
        high_card = unique_vals > 50 and dtype == "object"

        print(f"\nColumn: {col}")
        print(f"  Type: {dtype}")
        print(f"  Missing: {missing_vals}")
        print(f"  Unique: {unique_vals}")
        if constant:
            print("  ⚠️ Constant column (may be dropped)")
        if high_card:
            print("  ⚠️ High cardinality (check if meaningful)")
        if missing_vals > 0:
             print("  ⚠️ Missing values exist")
