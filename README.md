# Student Dropout Prediction

This project trains a simple classification model to predict whether a student is likely to drop out based on a small set of academic and demographic features.

## What it does
- Trains a Random Forest classifier on a synthetic dataset
- Saves the trained model to a local pickle file
- Provides a browser-based interface for prediction

## Features used
- GPA
- Attendance
- Study hours
- Family income
- Extracurricular participation
- Previous failures
- Age
- Internet access

## Run locally
1. Install dependencies:
   - `pip install scikit-learn numpy pytest streamlit`
2. Start the web app:
   - `streamlit run app.py`
3. Run the terminal script:
   - `python main.py`
4. Run tests:
   - `pytest -q`

## Web app
Open the local Streamlit URL shown in the terminal to interact with the student dropout predictor in a browser.
