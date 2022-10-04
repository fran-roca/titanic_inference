

def feature_engineering(data_df):

    """
        Function to encapsulate the variable engineering task

        Args:
           data_df (DataFrame):  Input dataset.

        Returns:
           DataFrame. Output dataset.
    """

    data_df = create_domain_knowledge_features(data_df)

    return data_df.copy()


def create_domain_knowledge_features(data_df):

    """
        Function to create context variables

        Args:
           df (DataFrame):  Dataset.
        Returns:
           DataFrame. Dataset.
    """

    data_df['Sex_child'] = 0
    data_df.loc[data_df.Age < 16, 'Sex_child'] = 1
    data_df.loc[data_df.Age < 16, 'Sex_male'] = 0
    data_df.loc[data_df.Age < 16, 'Sex_female'] = 0
    return data_df.copy()
