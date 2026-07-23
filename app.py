import streamlit as st
from main import load_model, predict_dropout

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓", layout="centered")
st.title("Student Dropout Predictor")
st.write("Enter a student's profile to estimate the likelihood of dropout.")

model = load_model()

with st.form("student_form"):
    gpa = st.slider("GPA", 0.0, 4.0, 2.5, 0.1)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    study_hours = st.slider("Study hours per week", 0, 40, 10)
    family_income = st.slider("Family income", 0, 200000, 40000, 1000)
    extracurricular = st.checkbox("Participates in extracurricular activities")
    previous_failures = st.slider("Previous failures", 0, 5, 0)
    age = st.slider("Age", 15, 30, 18)
    internet_access = st.checkbox("Has internet access")

    submitted = st.form_submit_button("Predict")

if submitted:
    student = {
        "gpa": gpa,
        "attendance": attendance,
        "study_hours": study_hours,
        "family_income": family_income,
        "extracurricular": 1 if extracurricular else 0,
        "previous_failures": previous_failures,
        "age": age,
        "internet_access": 1 if internet_access else 0,
    }

    result = predict_dropout(student, model)
    probability = result["probability"]
    label = result["label"]

    st.subheader("Prediction")
    if label == "dropout":
        st.error(f"High risk of dropout ({probability:.1%})")
    else:
        st.success(f"Low risk of dropout ({probability:.1%})")

    st.metric("Prediction", label.replace("_", " ").title())
    st.progress(min(max(probability, 0.0), 1.0))
