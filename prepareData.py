import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

source = 'Source'
def removeTargetValueFromFeatureNames(names, target):
    if target in names:
        names.remove(target)
        
def get_valid_values_index(data, missing):
    result_index = (data != missing[0])
    for i in range(1, len(missing)):
        result_index = result_index & (data != missing[i]) 
    return result_index

def clean_train_df_index(df, missing_values, target):
    result = get_valid_values_index(df[target], missing_values[target])
    for key in missing_values:
        if key!=target:
            res = get_valid_values_index(df[key], missing_values[key])
        result = result & res
    return result

def extract_target_variable(df, target):
    return df[target]


def separateTrainAndMissedDF(original_df, categorial_feature_names, numeric_features_names, missing_values, target):
    removeTargetValueFromFeatureNames(categorial_feature_names, target)
    removeTargetValueFromFeatureNames(numeric_features_names, target)

    valid_value_index = get_valid_values_index(original_df[target], missing_values[target])
    invalid_value_index = np.invert(valid_value_index)
    
    train_df = original_df[valid_value_index]
    df_with_missing_target = original_df[invalid_value_index]

    clean_train_df = train_df[clean_train_df_index(train_df, missing_values, target)]

    target_df = extract_target_variable(clean_train_df, target)
    train_df_without_target = without_target_variable(clean_train_df, target)
    df_with_missing_target_without_target = without_target_variable(df_with_missing_target, target)

    train_df_without_target[source] = 'Train'
    df_with_missing_target_without_target[source] = 'Calc'

    all_df = pd.concat([train_df_without_target, df_with_missing_target_without_target])

    numeric_df = extract_numeric_features(all_df, numeric_features_names)
    with warnings.catch_warnings(record=True):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df))
        scaled_numeric_df.index = numeric_df.index
        scaled_numeric_df.columns = numeric_df.columns
    
    categorical_df = all_df[categorial_feature_names]
    enc = OneHotEncoder(sparse=False)
    encoded_array = enc.fit_transform(categorical_df)
    encoded_df = pd.DataFrame(encoded_array)
    encoded_df.columns = enc.get_feature_names()
    encoded_df.index = categorical_df.index

    all_prepared_df = pd.merge(scaled_numeric_df, encoded_df, how='left', left_index=True, right_index=True)
    final_df = pd.merge(all_df[source].to_frame(), all_prepared_df, how='left', left_index=True, right_index=True)

    train_indexes = get_valid_values_index(final_df[source], ['Calc','Xzzzzz'])
    final_train_df = final_df[train_indexes]
    final_predict_df = final_df[np.invert(train_indexes)]
    del final_train_df[source]
    del final_predict_df[source]
    return final_train_df,final_predict_df, target_df
def without_target_variable(df, target):
    return df.drop(target, axis=1)

def extract_numeric_features(df, numeric_features_names):
    return df[numeric_features_names]
