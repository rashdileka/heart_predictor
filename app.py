# app.py

from flask import Flask, render_template, request
from model import predict_heart_disease

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form as floats
        input_features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
    except ValueError:
        return render_template('index.html', prediction="Please enter valid numbers.")

    prediction = predict_heart_disease(input_features)
    result_text = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected."

    return render_template('index.html', prediction=result_text)

if __name__ == '__main__':
    app.run(debug=True)
