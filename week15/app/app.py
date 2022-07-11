#------------------------------------------------------------------------------
# To run your web server, open up your terminal / command prompt
# and type:
#    cd <path to this file>
#    python practical-03c-deployment.py
#
#------------------------------------------------------------------------------

from flask import Flask, flash, request, redirect, url_for, Response
import requests
import os
import json
import tensorflow 
import tensorflow.keras as keras
import numpy as np

# Configure our application 
#
model_dir = 'activity_model'

# Initialize our Flask app.
# NOTE: Flask is used to host our app on a web server, so that
# we can call its functions over HTTP/HTTPS.
#
#app = Flask(__name__)
labels = ["JUMPING", "JUMPING_JACKS", "BOXING", "WAVING_2HANDS", "WAVING_1HAND", "CLAPPING_HANDS"]

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

model = keras.models.load_model(model_dir)

@app.route('/predict', methods=['POST'])
def predict():
    print('receiving keypoints')
    json_data = request.get_json()
    x_str = json_data['instances']
    X = np.array(x_str)
    pred = model(X).numpy()
    print(pred[0])
    index = np.argmax(pred[0])
    if pred[0][index] < 0.6:
        activity = "UNKNOWN"
    else:
        activity = labels[index]
    return Response(activity)



#------------------------------------------------------------------------------
# This starts our web server.
# Although we are running this on our local machine,
# this can technically be hosted on any VM server in the cloud!
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Only for debugging while developing

    app.run(host="0.0.0.0", debug=True, port=80)


