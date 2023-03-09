'''
Utils.py contains all utility functions
used during the inference process
'''

import pandas as pd
import numpy as np
from constants import *
from sklearn import preprocessing


def get_inference_data():

    '''
    Method for loading inference data
    Input: No input
    Output: Returns inference data features and labels
    Example usage: inference_data, labels = get_inference_data()
    '''

    data = pd.read_csv("Data\inference_heart_disease.csv")
    data.drop_duplicates(subset=None, inplace=True)
    data.duplicated().any()
    return data[data.columns.drop('target')], data['target']


def encode_features(df, features):

    '''
    Method for one-hot encoding all selected categorical fields
    Input: The method takes pandas dataframe and
    list of the feature names as input
    Output: Returns a dataframe with one-hot encoded features
    Example usage:
    one_hot_encoded_df = encode_features(dataframe, list_features_to_encode)
    '''

    encoded_df = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
    placeholder_df = pd.DataFrame()

    for f in features:
        if (f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')

    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in df.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    return encoded_df


def data_normalization(df):

    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''

    val = df.values
    min_max_normalizer = preprocessing.MinMaxScaler()
    norm_val = min_max_normalizer.fit_transform(val)
    df2 = pd.DataFrame(norm_val)
    return df2


def apply_pre_processing(data):

    '''
    Apply all pre-processing methods together
    Input: The method takes the inference data as pandas dataframe as input
    Output: Returns a dataframe after applying all preprocessing steps
    Example usage:
    processed_data = apply_pre_processing(df)
    '''

    feature_to_encode = FEATURE_TO_ENCODE
    encoded = encode_features(data, feature_to_encode)
    processed_data = data_normalization(encoded)

    return processed_data
