from urllib import response
from flask import Flask, request, jsonify, render_template
from model import sc
import numpy as np
import keras
import requests

app = Flask(__name__)
## load tensorflow model
model = keras.models.load_model("model.h5")

# Displays the home page on loading
@app.route("/")
def Home():
    return render_template("index.html")

# Function that does the prediction and gives results to the result page
@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    features = sc.transform(features)
    prediction = model.predict(features)
    prediction=prediction.item()

    age=request.form.get('age')
    cp=request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol= request.form.get('chol')

    return render_template("results.html", age=age, cp=cp, chol=chol, 
                             trestbps=trestbps, prediction=prediction) 

# Function that does search of the OpenMRS database using REST calls
@app.route("/search", methods = ["GET"])
def search():
    searchString=request.args.get('searchString')
    print("Got serach string", searchString)
    user='admin'
    password='Admin123'

    url=f"http://localhost:8080/openmrs/ws/rest/v1/patient?q={searchString}&v=default&limit=1"

    response= requests.get(url, auth=(user,password))
    jsonResponse= response.json()
    person=jsonResponse['results'][0]['person']

    person_age=person['age']
    person_gender=person['gender']
    person_name=person['display']

    person_gender=1 if person_gender=='M' else 0

    return render_template('index2.html',age=person_age,sex=person_gender,
    name=person_name, searchString=searchString)


if __name__ == "__main__":
    app.run(debug=True)

