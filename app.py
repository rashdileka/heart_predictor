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


import fitz  # PyMuPDF
from flask import make_response
from datetime import datetime
import os

@app.route('/download-report')
def download_report():
    input_data = session.get('input_data')
    if not input_data:
        return redirect(url_for('home'))

    name = session.get('name', 'N/A')
    age = session.get('age', 'N/A')
    sex = int(session.get('sex', 0))
    gender = "Male" if sex == 1 else "Female"
    advices = generate_advice(input_data)
    diagnosis = "Heart disease has been diagnosed."
    date_str = datetime.now().strftime("%B %d, %Y")

    # Create new PDF
    doc = fitz.open()
    page = doc.new_page()

    # Insert watermark (optional)
    image_path = os.path.join("static", "images", "watermark.png")
    if os.path.exists(image_path):
        try:
            img_rect = fitz.Rect(150, 200, 450, 500)
            page.insert_image(img_rect, filename=image_path, overlay=False, keep_proportion=True)
        except Exception as e:
            print(f"⚠️ Watermark insertion failed: {e}")

    # Header
    page.insert_text((50, 50), "🩺 MEDICAL REPORT", fontsize=16, fontname="helv", render_mode=3)
    page.insert_text((400, 50), f"Date: {date_str}", fontsize=10)

    # Patient Info
    box_top = 80
    page.draw_rect(fitz.Rect(50, box_top, 550, box_top + 60), color=(0, 0, 0), width=1)
    page.insert_text((60, box_top + 10), f"👤 Name   : {name}", fontsize=11)
    page.insert_text((60, box_top + 25), f"🎂 Age    : {age} years", fontsize=11)
    page.insert_text((60, box_top + 40), f"🚻 Gender : {gender}", fontsize=11)

    # Diagnosis
    diag_top = box_top + 80
    page.draw_rect(fitz.Rect(50, diag_top, 550, diag_top + 40), color=(0, 0, 0), width=1)
    page.insert_text((60, diag_top + 12), f"🧪 Diagnosis: {diagnosis}", fontsize=11)

    # Advice
    advice_top = diag_top + 60
    page.insert_text((50, advice_top), "🩺 Medical Advice:", fontsize=12, render_mode=3)

    for i, advice in enumerate(advices, 1):
        # Show only English part inside brackets
        english_only = advice.split('(')[-1].rstrip(')') if '(' in advice else advice
        y = advice_top + 20 + i * 16
        page.insert_text((60, y), f"{i}. {english_only}", fontsize=10)

    # Signature
    signature_y = advice_top + 40 + len(advices) * 16
    page.insert_text((50, signature_y), "---------------------------------------", fontsize=11)
    page.insert_text((50, signature_y + 15), "Doctor's Signature", fontsize=10)

    # Footer
    footer_text = "📄 Created by Heart Disease Prediction Website"
    page.insert_text((50, 800 - 30), footer_text, fontsize=9, color=(0.4, 0.4, 0.4))

    # Return PDF
    pdf_bytes = doc.write()
    response = make_response(pdf_bytes)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=doctor_report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
