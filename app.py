from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

# 🏠 HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# 📊 ANALYSIS PAGE
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# 🏥 HOSPITAL PAGE
@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

# 🔮 PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(request.form['Age']),
        float(request.form['Fatigue']),
        float(request.form['Slowing']),
        float(request.form['Pain']),
        float(request.form['Hygiene']),
        float(request.form['Movement'])
    ]

    prediction = model.predict([features])
    prob = model.predict_proba([features])[0][1] * 100

    if prediction[0] == 1:
        result = "High Risk"
        suggestion = "Consult a psychiatrist immediately."
    else:
        result = "Low Risk"
        suggestion = "Maintain a healthy lifestyle."

    return render_template('analysis.html',
                           prediction_text=result,
                           probability=round(prob,2),
                           suggestion=suggestion)

if __name__ == "__main__":
    app.run(debug=True)