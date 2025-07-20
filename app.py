from flask import Flask, render_template, request
from model import predict_heart_disease  # make sure this function accepts list input and returns 0 or 1

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form, convert to proper types
        input_features = [
            int(request.form['age']),
            int(request.form['sex']),
            int(request.form['cp']),
            int(request.form['trestbps']),
            int(request.form['chol']),
            int(request.form['fbs']),
            int(request.form['restecg']),
            int(request.form['thalach']),
            int(request.form['exang']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thal'])
        ]
    except (ValueError, KeyError):
        # If any input missing or wrong type
        return render_template('index.html', prediction="Please enter valid inputs for all fields.")

    try:
        # Call your prediction function with input features list
        prediction = predict_heart_disease(input_features)
        # Interpret model output
        result_text = "Heart Disease Detected!" if prediction == 1 else "No Heart Disease Detected."
    except Exception as e:
        # Handle any errors during prediction (log e if you want)
        result_text = "An error occurred during prediction. Please try again."

    return render_template('index.html', prediction=result_text)

if __name__ == '__main__':
    app.run(debug=True)
