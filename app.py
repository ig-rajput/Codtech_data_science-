
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        final_features = [np.array(input_features)]
        prediction = model.predict(final_features)
        output = "Malignant" if prediction[0] == 0 else "Benign"
        return render_template("index.html", prediction_text=f"Prediction: {output}")
    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid input.")

if __name__ == "__main__":
    app.run(debug=True)
