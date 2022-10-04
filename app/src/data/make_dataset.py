import pandas as pd
from ..features.feature_engineering import feature_engineering
from app import cos, init_cols


def make_dataset(data, model_info, cols_to_remove, model_type='RandomForest'):

    """
        Function to create the dataset used for training
        of the model.

        Args:
           data (List):  List with observation arrived by request.
           model_info (dict):  Production model information.

        Kwargs:
           model_type (str): type of model used.

        Returns:
           DataFrame. Dataset to infer.
    """

    print('---> Getting data')
    data_df = get_raw_data_from_request(data)
    print('---> Transforming data')
    data_df = transform_data(data_df, model_info, cols_to_remove)
    print('---> Feature engineering')
    data_df = feature_engineering(data_df)
    print('---> Preparing data for training')
    data_df = pre_train_data_prep(data_df, model_info)

    return data_df.copy()


def get_raw_data_from_request(data):

    """
        Function to get new observations from request

        Args:
           data (List):  List with the observation arrived by request.

        Returns:
           DataFrame. Dataset with the input data.
    """
    for key in init_cols:
        if key not in data.keys():
            data[key] = None

    try:
        return (pd.DataFrame(data) if len(data)==len(init_cols) else None)
    except Exception:
        return pd.DataFrame(data, index=[0])


def transform_data(data_df, model_info, cols_to_remove):
    """
        Function that allows performing the first transformation tasks
         of the input data.

        Args:
            data_df (DataFrame):  Input data set.
            model_info (dict):  Information of the model in production.
            cols_to_remove (list): Columns to remove.

        Returns:
           DataFrame. Transformed dataset.
    """

    print('------> Removing unnecessary columns')
    data_df = remove_unwanted_columns(data_df, cols_to_remove)

    data_df['Pclass'] = data_df['Pclass'].astype(str)

    # creating original dummies
    print('------> Encoding data')
    print('---------> Getting encoded columns from cos')
    enc_key = model_info['objects']['encoders']+'.pkl'
    # getting the columns present in the training from COS
    enc_cols = cos.get_object_in_cos(enc_key)
    # dummy columns generated in the input data
    data_df = pd.get_dummies(data_df)

    # adding the missing dummy columns in the input data
    data_df = data_df.reindex(columns=enc_cols, fill_value=0)

    return data_df.copy()


def pre_train_data_prep(data_df, model_info):

    """
        Function that performs the last transformations on the data
        before training (null imputation)

        Args:
            data_df (DataFrame):  Input dataset.
            model_info (dict):  Production model information.

        Returns:
            DataFrame. Output dataset.
    """

    print('------> Getting imputer from cos')
    imputer_key = model_info['objects']['imputer']+'.pkl'
    data_df = input_missing_values(data_df, imputer_key)

    return data_df.copy()


def input_missing_values(data_df, key):

    """
        Null imputation function

        Args:
            data_df (DataFrame):  Input dataset.
            key (str):  Name of the imputing object in COS.

        Returns:
            DataFrame. Output dataset.
    """

    print('------> Inputing missing values')
    # we get the SimpleImputer object from COS
    imputer = cos.get_object_in_cos(key)
    data_df = pd.DataFrame(imputer.transform(data_df), columns=data_df.columns)

    return data_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Function to remove unnecessary variables

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)





