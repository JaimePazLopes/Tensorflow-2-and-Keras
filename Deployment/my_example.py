from flask import Flask, request, jsonify



import numpy as np
from tensorflow.keras.models import load_model
import joblib


def return_prection(model, scaler, sample_json):

  classes = np.array(['setosa', 'versicolor', 'virginica'])

  s_len = sample_json["sepal_length"]
  s_wid = sample_json["sepal_width"]
  p_len = sample_json["petal_length"]
  p_wid = sample_json["petal_width"]

  flower = scaler.transform([[s_len, s_wid, p_len, p_wid]])

  clas_ind = model.predict_classes(flower)

  return classes[clas_ind[0]]



app = Flask(__name__)

@app.route("/")
def index():
	return "<h1>FLASK APP IS RUNNING</h1>"


flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")


@app.route("/api/flower", methods=["POST"])
def flower_prediction():
	content = request.json

	results = return_prection(flower_model, flower_scaler, content)

	return jsonify(results)


if __name__ == "__main__":
	app.run()