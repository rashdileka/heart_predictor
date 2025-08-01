from flask import Flask, render_template, request, redirect, url_for, session
from model import predict_heart_disease

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for sessions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
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
    except:
        return render_template('index.html', prediction="කරුණාකර සියලුම තොරතුරු නිවැරදිව ඇතුළත් කරන්න. (Please enter all information correctly)")

    prediction = predict_heart_disease(input_data)

    if prediction == 1:
        # Save input data to session to use in advice page
        session['input_data'] = input_data
        return redirect(url_for('advice'))
    else:
        return render_template('index.html', prediction="හෘද රෝගයක් හඳුනා ගැනීම් නොවී ඇත. (No heart disease has been diagnosed)")

@app.route('/advice')
def advice():
    input_data = session.get('input_data')
    if not input_data:
        return redirect(url_for('home'))  # No input data, go home

    advices = generate_advice(input_data)
    return render_template('result.html', prediction="හෘද රෝගයක් හඳුනා ගන්නා ලදී! (Heart disease was diagnosed)", advices=advices)

def generate_advice(data):
    [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal] = data
    advice = []

    if age > 50:
        advice.append("ඔබගේ වයස අනුව වෛද්‍ය පරීක්ෂණ සාමාන්‍ය ලෙස සිදු කළ යුතුය. (Medical checkups should be done regularly depending on your age)")
    if trestbps > 130:
        advice.append("ඔබගේ රුධිර පීඩනය උසස් වන නිසා ලුණු පරිභෝජනය අඩු කරන්න. (Reduce your salt intake as it can raise your blood pressure)")
    if chol > 200:
        advice.append("කොලෙස්ටරෝල් මට්ටම ඉහල බැවින් කෙටි තෙල් සහ පිඟාන ආහාර වලින් වලකින්න. (Avoid fried foods and fatty foods as they can increase cholesterol levels.)")
    if fbs == 1:
        advice.append("රුධිර සීනි උසස් බැවින්, සීනි අඩු ආහාර වලින් පෝෂණය වන්න. (Since blood sugar is high, eat foods low in sugar)")
    if exang == 1:
        advice.append("වයාමයට ප්‍රතික්‍රියා ලෙස වේදනාවක් ඇතිවී ඇත්නම් වෛද්‍ය උපදෙස් ලබා ගන්න. (Seek medical advice if you experience pain as a reaction to exercise)")
    if oldpeak > 2:
        advice.append("ST Depression මට්ටම අධික බැවින් වෛද්‍ය පරීක්ෂණ සඳහා යොමු වන්න. (Since your ST depression level is high, seek medical attention)")
    if thalach < 100:
        advice.append("හෘදගාමී ව්‍යායාමයක් අනුගමනය කරන්න. (Follow a cardio workout)")
    if cp in [1, 2]:
        advice.append("ඔබට උරහිස් වේදනාවක් ඇතිවී ඇති නිසා වෛද්‍ය උපදෙස් ලබා ගැනීම වැදගත්ය. (It is important to seek medical advice since you have shoulder pain)")

    if len(advice) == 0:
        advice.append("ඔබට සාමාන්‍ය ව්‍යායාම, සයිනික් ආහාර හා වෛද්‍ය උපදෙස් ලබා ගැනීම වැදගත්ය. (It is important for you to get regular exercise, a healthy diet, and medical advice)")

    return advice

if __name__ == '__main__':
    app.run(debug=True)
