import pickle
from flask import Flask, request
import pandas as pd
import flasgger
from flasgger import Swagger
import numpy as numpy

app = Flask(__name__)
Swagger(app)

pickle_in = open("finalized_model.pickle", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():
    """Let's Authenticate the Cardiovascular Disease
    This is using docstrings for specifications.
    ---
    parameters:
      - name: age
        in: query
        type: number
        required: true
      - name: gender
        in: query
        type: number
        required: true
      - name: height
        in: query
        type: number
        required: true
      - name: weight
        in: query
        type: number
        required: true
      - name: ap_hi
        in: query
        type: number
        required: true
      - name: ap_lo
        in: query
        type: number
        required: true
      - name: cholesterol	
        in: query
        type: number
        required: true
      - name: gluc	
        in: query
        type: number
        required: true
      - name: smoke	
        in: query
        type: number
        required: true
      - name: alco
        in: query
        type: number
        required: true
      - name: active	
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    age = request.args.get("age")
    gender = request.args.get("gender")
    height = request.args.get("height")
    weight = request.args.get("weight")
    ap_hi = request.args.get("ap_hi")
    ap_lo = request.args.get("ap_lo")
    cholesterol = request.args.get("cholesterol")
    gluc = request.args.get("gluc")
    smoke = request.args.get("smoke")
    alco = request.args.get("alco")
    active = request.args.get("active")

    prediction = classifier.predict(
        [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    print(prediction)
    if prediction[0] == 0:
        pickle_zero = open("random_forest_0.pickle", "rb")
        classifier_zero = pickle.load(pickle_zero)
        prediction_final = classifier_zero.predict(
            [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        print(prediction_final)

    if prediction[0] == 1:
        pickle_first = open("random_forest_1.pickle", "rb")
        classifier_first = pickle.load(pickle_first)
        prediction_final = classifier_first.predict(
            [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        print(prediction_final)

    if prediction[0] == 2:
        pickle_second = open("random_forest_2.pickle", "rb")
        classifier_second = pickle.load(pickle_second)
        prediction_final = classifier_first.predict(
            [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        print(prediction_final)

        if prediction[0] == 3:
            pickle_three = open("random_forest_3.pickle", "rb")
            classifier_three = pickle.load(pickle_three)
            prediction_final = classifier_three.predict(
                [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
            print(prediction_final)

    if prediction[0] == 4:
        pickle_four = open("logistic_model4.pickle", "rb")
        classifier_four = pickle.load(pickle_four)
        prediction_final = classifier_four.predict(
            [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        print(prediction_final)

    return "Hello The answer is "+str(prediction_final[0])
