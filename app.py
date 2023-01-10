import pickle
from flask import Flask, request, app, jsonify,url_for,render_template
import numpy as np
import pandas as pd

# here we define the Flask app
app=Flask(__name__) 

# load the model
model = pickle.load(open('./model/model_v_0_1.pkl','rb'))
scalar = pickle.load(open('./model/scaler_v_0_1.pkl','rb'))


@app.route('/')
def home():

	return render_template('home.html')

# for post request
@app.route('/predict_api', methods = ['POST'])
def predict_api():
	data = request.get_json(force=True)['data']
	print(data)
	# data = request.json['data']
	# print(data)
	print(list(data.values()))
	print(np.array(list(data.values())).reshape(1,-1))
	new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
	output = model.predict(new_data)
	print("ouput:",output[0])
	return jsonify(output[0])


if __name__ == "__main__":
	app.run(debug=True)