from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Scale numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prediction = model.predict(input_df)[0]

    result = "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"

    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)