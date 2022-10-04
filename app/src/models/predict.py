from ..data.make_dataset import make_dataset
from app import cos, client
from cloudant.query import Query


def predict_pipeline(data, model_info_db_name='titanic_db'):

    """
        Function to manage the complete inference pipeline
        of the model.

        Args:
            path (str):  Data path.

        Kwargs:
            model_info_db_name (str):  database to use to store
            the model info.

        Returns:
            list. List with predictions made.
    """

    # Loading training settings
    model_config = load_model_config(model_info_db_name)['model_config']
    # columns to remove
    cols_to_remove = model_config['cols_to_remove']

    # get the information of the model in production
    model_info = get_best_model_info(model_info_db_name)
    # loading and transforming the input data
    data_df = make_dataset(data, model_info, cols_to_remove)

    # Downloading the model object
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)

    # performing the inference with the input data
    return model.predict(data_df).tolist()


def load_model(name, bucket_name='deposittitanic'):
    """
         Function to load the model into IBM COS

         Args:
             name (str):  Object name in COS to load.

         Kwargs:
             bucket_name (str):  IBM COS repository to use.

        Returns:
            obj. Downloaded object.
     """
    return cos.get_object_in_cos(name, bucket_name)


def get_best_model_info(db_name):
    """
         Function to load the IBM Cloudant model info

         Args:
             db_name (str):  Database name.

         Kwargs:
             bucket_name (str):  IBM COS repository to use.

        Returns:
            dict. Model info.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]


def load_model_config(db_name):
    """
        Function to load model info from IBM Cloudant.

        Args:
            db_name (str):  Database name.

        Returns:
            dict. Document with model configuration.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'titanic_config'}})
    return query()['docs'][0]
