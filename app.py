# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:23:02 2020

@author: Sanket Kotkar

File : app.py
"""
#%%
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
Model = pickle.load(open('Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Template.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Model.predict(final_features)

    output = round(prediction[0])
    predict = ""
    if output == 1:
        predict = 'Yes'
    else:
        predict = 'No'

    return render_template('Template.html', prediction_text='{}'.format(predict))


if __name__ == "__main__":
    app.run(debug=True)
