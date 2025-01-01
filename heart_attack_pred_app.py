import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for
import pickle
import os

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("heart.html")


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 15)
    loaded_model = pickle.load(open("heart_attack_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/result", methods=["POST"])
def result():
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)
    if int(result) == 1:
        prediction = "High chances of Heart Attack"
    else:
        prediction = "Heart Attack not likely"
    return render_template("result.html", prediction=prediction)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)
            predictions = bulk_predict(data)
            result_file_path = os.path.join("uploads", "predictions.csv")
            predictions.to_csv(result_file_path, index=False)
            return send_file(result_file_path, as_attachment=True)
    return redirect(url_for("home"))


def bulk_predict(data):
    loaded_model = pickle.load(open("heart_attack_model.pkl", "rb"))
    predictions = loaded_model.predict(data)
    data["prediction"] = predictions
    return data


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
