import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_cleaned_encoded_data(file_path, types_dict):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, header=0)
    # change_types(df, types_dict) # redundant
    __encode_categorical_features(df)
    cleaned_data = __handle_missing_data(df)
    return cleaned_data


def split_train_test(df, frac):
    return train_test_split(df, test_size=frac, random_state=42)


def __change_types(data_frame, types):
    for col_name, t in types.items():
        if t == 'Categorical':
            data_frame[col_name] = data_frame[col_name].astype('category')
        if t == 'Class':
            data_frame[col_name] = data_frame[col_name].astype('object')


def __encode_categorical_features(df: pd.DataFrame):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype not in [np.float64, np.int64]:  # numeric
            df[col] = le.fit_transform(df[col])


def __handle_missing_data(df):
    cleaned_df = df  # check if its by ref or it copy
    if cleaned_df.empty:
        return None
    none_column_data = cleaned_df.columns[cleaned_df.isnull().any()]
    for col in none_column_data:
        col_type = cleaned_df[col].dtype
        if col_type in [np.float64, np.int64]:  # numeric
            cleaned_df.loc[:, col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        else:  # categorical value
            cleaned_df.loc[:, col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
    return cleaned_df
