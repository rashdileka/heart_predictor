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
        # Save name, gender, and age in session
        session['name'] = request.form['name']
        session['gender'] = request.form['sex']
        session['age'] = request.form['age']
        # Save all other inputs in session
        session['cp'] = request.form['cp']
        session['trestbps'] = request.form['trestbps']
        session['chol'] = request.form['chol']
        session['fbs'] = request.form['fbs']
        session['restecg'] = request.form['restecg']
        session['thalach'] = request.form['thalach']
        session['exang'] = request.form['exang']
        session['oldpeak'] = request.form['oldpeak']
        session['slope'] = request.form['slope']
        session['ca'] = request.form['ca']
        session['thal'] = request.form['thal']

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
        return render_template('index.html',
                               prediction="කරුණාකර සියලුම තොරතුරු නිවැරදිව ඇතුළත් කරන්න. (Please enter all information correctly)")

    prediction = predict_heart_disease(input_data)

    if prediction == 1:
        session['input_data'] = input_data
        return redirect(url_for('advice'))
    else:
        return render_template('index.html',
                               prediction="හෘද රෝගයක් හඳුනා ගැනීම් නොවී ඇත. (No heart disease has been diagnosed)")


@app.route('/advice')
def advice():
    input_data = session.get('input_data')
    if not input_data:
        return redirect(url_for('home'))  # No input data, go home

    advices = generate_advice(input_data)
    return render_template('result.html', prediction="හෘද රෝගයක් හඳුනා ගන්නා ලදී! (Heart disease was diagnosed)",
                           advices=advices)


def generate_advice(data):
    [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal] = data
    advice = []

    if age > 50:
        advice.append(
            "ඔබගේ වයස අනුව වෛද්‍ය පරීක්ෂණ සාමාන්‍ය ලෙස සිදු කළ යුතුය. (Medical checkups should be done regularly depending on your age)")
    if trestbps > 130:
        advice.append(
            "ඔබගේ රුධිර පීඩනය උසස් වන නිසා ලුණු පරිභෝජනය අඩු කරන්න. (Reduce your salt intake as it can raise your blood pressure)")
    if chol > 200:
        advice.append(
            "කොලෙස්ටරෝල් මට්ටම ඉහල බැවින් කෙටි තෙල් සහ පිඟාන ආහාර වලින් වලකින්න. (Avoid fried foods and fatty foods as they can increase cholesterol levels.)")
    if fbs == 1:
        advice.append(
            "රුධිර සීනි උසස් බැවින්, සීනි අඩු ආහාර වලින් පෝෂණය වන්න. (Since blood sugar is high, eat foods low in sugar)")
    if exang == 1:
        advice.append(
            "වයාමයට ප්‍රතික්‍රියා ලෙස වේදනාවක් ඇතිවී ඇත්නම් වෛද්‍ය උපදෙස් ලබා ගන්න. (Seek medical advice if you experience pain as a reaction to exercise)")
    if oldpeak > 2:
        advice.append(
            "ST Depression මට්ටම අධික බැවින් වෛද්‍ය පරීක්ෂණ සඳහා යොමු වන්න. (Since your ST depression level is high, seek medical attention)")
    if thalach < 100:
        advice.append("හෘදගාමී ව්‍යායාමයක් අනුගමනය කරන්න. (Follow a cardio workout)")
    if cp in [1, 2]:
        advice.append(
            "ඔබට උරහිස් වේදනාවක් ඇතිවී ඇති නිසා වෛද්‍ය උපදෙස් ලබා ගැනීම වැදගත්ය. (It is important to seek medical advice since you have shoulder pain)")

    if len(advice) == 0:
        advice.append(
            "ඔබට සාමාන්‍ය ව්‍යායාම, සයිනික් ආහාර හා වෛද්‍ය උපදෙස් ලබා ගැනීම වැදගත්ය. (It is important for you to get regular exercise, a healthy diet, and medical advice)")

    return advice


import fitz  # PyMuPDF
from flask import make_response, redirect, url_for, session
from datetime import datetime
import os

@app.route('/download-report')
def download_report():
    input_data = session.get('input_data')
    if not input_data:
        return redirect(url_for('home'))  # No input data

    # Extract patient session data
    name = session.get('name', 'N/A')
    age = session.get('age', 'N/A')
    sex = int(session.get('sex', 0))
    gender = "Male" if sex == 1 else "Female"
    cp = session.get('cp', 'N/A')
    trestbps = session.get('trestbps', 'N/A')
    chol = session.get('chol', 'N/A')
    fbs = session.get('fbs', 'N/A')
    restecg = session.get('restecg', 'N/A')
    thalach = session.get('thalach', 'N/A')
    exang = session.get('exang', 'N/A')
    oldpeak = session.get('oldpeak', 'N/A')
    slope = session.get('slope', 'N/A')
    ca = session.get('ca', 'N/A')
    thal = session.get('thal', 'N/A')

    advices = generate_advice(input_data)
    diagnosis = "Heart disease has been diagnosed." if predict_heart_disease(input_data) == 1 else "No heart disease detected."
    date_str = datetime.now().strftime("%B %d, %Y")

    # Create PDF
    doc = fitz.open()
    page = doc.new_page()

    # Watermark
    image_path = os.path.join("static", "images", "watermark.png")
    if os.path.exists(image_path):
        try:
            watermark_rect = fitz.Rect(180, 200, 420, 480)
            page.insert_image(watermark_rect, filename=image_path, overlay=True, keep_proportion=True)
        except Exception as e:
            print(f"⚠️ Watermark insertion failed: {e}")

    # Header
    page.insert_text((50, 40), "🩺 HEART DISEASE REPORT", fontsize=17, render_mode=3, fontname="helv")
    page.insert_text((400, 40), f"📅 Date: {date_str}", fontsize=10, fontname="helv")

    # Patient Details Box
    y = 80
    page.draw_rect(fitz.Rect(50, y, 550, y + 60), color=(0.2, 0.2, 0.2), width=1)
    page.insert_text((60, y + 10), f"👤 Name   : {name}", fontsize=11)
    page.insert_text((60, y + 25), f"🎂 Age    : {age} years", fontsize=11)
    page.insert_text((60, y + 40), f"🚻 Gender : {gender}", fontsize=11)

    # Input Data
    y += 80
    entries = [
        ("💓 Chest Pain Type", cp),
        ("💉 Resting BP", f"{trestbps} mm Hg"),
        ("🧴 Cholesterol", f"{chol} mg/dL"),
        ("🥤 Fasting Blood Sugar > 120", "Yes" if fbs == 1 else "No"),
        ("📈 Resting ECG", restecg),
        ("🏃 Max Heart Rate", f"{thalach} bpm"),
        ("🏋️ Exercise-Induced Angina", "Yes" if exang == 1 else "No"),
        ("📉 ST Depression (Oldpeak)", f"{oldpeak} mm"),
        ("🔺 Slope of ST Segment", slope),
        ("💉 Major Vessels Colored", ca),
        ("🩸 Thalassemia", thal)
    ]

    for label, value in entries:
        page.insert_text((60, y), f"{label}: {value}", fontsize=11)
        y += 18

    # Diagnosis Section
    y += 10
    page.draw_rect(fitz.Rect(50, y, 550, y + 40), color=(0.2, 0.2, 0.2), width=1)
    page.insert_text((60, y + 12), f"🧪 Diagnosis: {diagnosis}", fontsize=11, fontname="helv")
    y += 60

    # Advice Section
    page.insert_text((50, y), "🩺 Doctor's Advice:", fontsize=12, render_mode=3)
    for i, advice in enumerate(advices, start=1):
        clean_advice = advice.split("(")[-1].rstrip(")") if "(" in advice else advice
        y += 16
        page.insert_text((60, y), f"{i}. {clean_advice}", fontsize=10)

    # Footer
    footer = "📄 Generated by Heart Disease Prediction System • Confidential Medical Report"
    page.insert_text((50, 820), footer, fontsize=9, color=(0.4, 0.4, 0.4))

    # Return PDF
    pdf_bytes = doc.write()
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=heart_disease_report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
