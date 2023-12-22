from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("your_model.pkl")

# Load the preprocessor
preprocessor = joblib.load("model_preprocessor.pkl")

# Function to get precautions based on predicted disease
def get_precautions(predicted_disease):
    if predicted_disease == 'Heart_Disease':
        return [
            "Precaution 1: Adopt a heart-healthy lifestyle by incorporating a balanced diet rich in fruits, vegetables, and whole grains. Limit saturated and trans fats, cholesterol, and sodium intake.",
            "Precaution 2: Engage in regular physical activity to maintain a healthy weight, strengthen the heart, and improve overall cardiovascular health.",
            "Precaution 3: Schedule regular check-ups with a healthcare professional to monitor blood pressure, cholesterol levels, and other relevant cardiovascular markers."
        ]
    elif predicted_disease == 'Diabetes':
        return [
            "Precaution 1: Manage and maintain a healthy weight through a combination of a well-balanced diet and regular physical activity.",
            "Precaution 2: Monitor blood glucose levels as recommended by healthcare professionals and adhere to prescribed medications or insulin regimens.",
            "Precaution 3: Educate oneself about diabetes management, including understanding the impact of diet, exercise, and stress on blood sugar levels."
        ]
    elif predicted_disease == 'Obesity':
        return [
            "Precaution 1: Establish and maintain a well-balanced diet that includes appropriate portion sizes and avoids excessive consumption of high-calorie, low-nutrient foods.",
            "Precaution 2: Engage in regular physical activity, combining both aerobic exercises and strength training, to support weight management.",
            "Precaution 3: Seek guidance from healthcare professionals or nutritionists for personalized weight loss strategies and lifestyle modifications."
        ]
    elif predicted_disease == 'Hypertension':
        return [
            "Precaution 1: Adopt a low-sodium diet by reducing the intake of processed foods and incorporating more fresh fruits, vegetables, and lean proteins.",
            "Precaution 2: Engage in regular aerobic exercise, such as brisk walking, swimming, or cycling, to help control blood pressure",
            "Precaution 3: Limit alcohol consumption and avoid tobacco products, as they can contribute to elevated blood pressure."
        ]
    elif predicted_disease == 'Healthy':
        return [
            "Recommendation 1: Continue maintaining a healthy lifestyle with a balanced diet, regular exercise, and proper stress management.",
            "Recommendation 2: Schedule routine health check-ups to monitor overall well-being and catch any potential health issues early.",
            "Recommendation 3: Stay informed about preventive healthcare measures and consider health screenings appropriate for one's age and risk factors."
        ]
    
    # Add similar conditions for other diseases (Hypertension, Obesity) or 'Healthy'

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# ... (other imports)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = {
            'Gender': [request.form['gender']],
            'Age': [int(request.form['age'])],
            'BMI': [float(request.form['bmi'])],
            'Smoking_Status': [request.form['smoking']],
            'BloodPressure_Systolic': [int(request.form['blood_pressure'])],
            'Diet': [request.form['diet']],
            'Alcohol_Consumption': [request.form['alcohol']],
            'Physical_Activity': [request.form['physical']],
            'Cholesterol': [request.form['chol']],
            'SleepDuration': [request.form['sleep']]
        }

        # Create a DataFrame from user input
        user_input_df = pd.DataFrame(user_input)

        # Apply the same preprocessing to user input
        user_input_encoded = preprocessor.transform(user_input_df)

        # Predict disease
        predicted_disease = model.predict(user_input_encoded)[0]

        # Get precautions based on predicted disease
        precautions = get_precautions(predicted_disease)

        # Enumerate the precautions for the template
        enumerated_precautions = list(enumerate(precautions, 1))
        
        # Debug information
        app.logger.info(f"Predicted Disease: {predicted_disease}")
        app.logger.info(f"Precautions: {precautions}")

        # Pass the prediction and enumerated precautions to the template
        return render_template('result.html', prediction=predicted_disease, precautions=enumerated_precautions)

if __name__ == '__main__':
    app.run(debug=True)
