# Dependencies
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import csv

# Your API definition
app = Flask(__name__)
# App functions that will be executed when an HTTP request(POST)
# is sent on localhost:port/predict 
@app.route('/predict', methods=['POST'])
# Predict process consist of the following processes:
#   1. Reception of the request
#   2. Input Dataframe is preprocessed with dummy values (as ML Model was created)
#   3. Logistic Regression model is applied to Input data to predict the survival of the person
#   4. This results list is then added to the Database.csv file and then is returned as JSON via the web-app
#   Note: the web-app will notify if the model is not detected, if this is the case you will need to rerun the model script and try again 
def predict():
    if lr:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            filename = open("Database.csv", "a", newline='')
            write = csv.writer(filename)
            write.writerows(map(lambda x:[x], prediction))
            filename.close()
            return jsonify({'prediction': str(prediction)})
    else:
        print ('Train the model first')
        return ('No model here to use')
# Model loading before web app running
if __name__ == '__main__':
    port = 5000
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(debug=True, host='0.0.0.0', port=port) # Web app is hosted in localhost:port
