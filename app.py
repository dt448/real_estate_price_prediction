import pickle
from flask import Flask, request, app, jsonify,url_for,render_template
import numpy as np
import pandas as pd

# here we define the Flask app
app=Flask(__name__) 

# load the model
model = pickle.load(open('./model/model_v_0_1.pkl','rb'))

@app.route('/')
def home():

	return render_template('home.html')

# for post request
@app.route('/predict_api', methods = ['POST'])
def predict_api():
	data = request.json['data']
	print(data)
	print(np.array(list(data.values())[0]).reshape(1,-1))