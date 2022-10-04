from flask import Flask, request
import os
from app.src.models.predict import predict_pipeline
import warnings

warnings.filterwarnings('ignore')

# -*- coding: utf-8 -*-
app = Flask(__name__)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))


# Using the decorator @app.route to manage routers
@app.route('/', methods=['GET'])
def root():
    """
        Function to manage the output of the root path.

        Returns:
           dict.  Output message
    """
    return {'Project':'Titanic - Inference'}


# route to run the inference pipeline (POST)
@app.route('/predict', methods=['POST'])
def predict_route():
    """
        Function to run the inference pipeline.

        Returns:
           dict.  Output message (prediction)
    """

    # Get the data passed by the request
    data = request.get_json()

    # Run inference pipeline
    y_pred = predict_pipeline(data)

    return {'Predicted value': y_pred}


# main
if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)
