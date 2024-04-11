# Dependencies
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flasgger import Swagger

# API definition
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://myuser:1234@db:5432/mydatabase'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
swagger = Swagger(app)

# Prediccion class is in charge of the definition of formatting data entries (predictions) to the DB (Hosted on Postgresql container)
class Prediccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.Boolean(5), unique=False, nullable=False)

    def __repr__(self):
        return '<Prediccion %r>' % self.prediction
    
# Flask app routing to localhost:port/predict
# Model loading for further prediction and storage of results
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict survival based on input data.

    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            Age:
              type: integer
              description: Age of the person
            Sex:
              type: string
              description: Sex of the person
            Embarked:
              type: string
              description: Port of embarkation
    responses:
      200:
        description: Prediction result
    """
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print('Model columns loaded')
    if lr:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_, index=[0]))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediccioncalc = list(lr.predict(query))
            dato_in=Prediccion(prediction=bool(prediccioncalc[0]))
            db.session.add(dato_in)
            db.session.commit()
            return jsonify({'prediction': str(prediccioncalc)})
    else:
        print('Train the model first')
        return ('No model here to use')

# App functions that will be executed when an HTTP request(POST)

# Predict process consist of the following processes
#   1. Reception of the request
#   2. Input Dataframe is preprocessed with dummy values (as ML Model was created)
#   3. Logistic Regression model is applied to Input data to predict the survival of the person
#   4. This results list is then added to the PostgreSQL and then is returned as JSON via the web-app
#   Note: the web-app will notify if the model is not detected, if this is the case you will need to rerun the model script and try again

if __name__ == '__main__':
    port = 5000
    app.run(debug=True, host='0.0.0.0', port=port) # Web app is hosted in localhost:port
