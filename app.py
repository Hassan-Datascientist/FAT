from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and feature names
model = joblib.load("diabetes_model.pkl")
columns = joblib.load("columns.pkl")

@app.route('/')
def home():
    return render_template("index.html", columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        values = [float(request.form[col]) for col in columns]
        input_data = np.array([values])

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        prob_percent = round(probability * 100, 2)

        return render_template("result.html", result=result, prob=prob_percent)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
