import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# App initialization
app = Flask(__name__)

# Load Models

with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

from tensorflow.keras.models import load_model
model_ann = load_model('titanic_model.h5')

# rooting
@app.route('/') # Homepage
def home():
    return '<h1> It works! </h1>'

@app.route('/predict', methods=['POST'])
def titanic_predict():
    args= request.json
    print(args,dir(args))

    data_inf = {
        'PassengerId': args.get('PassengerId'),
        'Pclass': args.get('Pclass'),
        'Name': args.get('Name'),
        'Sex': args.get('Sex'),
        'Age': args.get('Age'),
        'SibSp': args.get('SibSp'),
        'Parch': args.get('Parch'),
        'Ticket':args.get('Ticket'),
        'Fare': args.get('Fare'),
        'Cabin': args.get('Cabin'),
        'Embarked': args.get('Embarked')
    }

    print('[DEBUG] Data Inference:', data_inf)
    
    data_inf= pd.DataFrame([data_inf])
    data_inf_transform = model_pipeline.transform(data_inf)
    y_pred_inf = model_ann.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5,1,0)

    if y_pred_inf == 0:
        label = 'Not Survived'
    else:
        label = 'Survived'

    print('[DEBUG] result :', y_pred_inf, label)
    print('')

    response=jsonify(
        result = str(y_pred_inf),
        label_names=label
    )

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')