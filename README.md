# 🧠 Mental Health Prediction for Tech Workers

This project uses machine learning to predict the likelihood of mental health disorders among tech workers, using the OSMI (Open Sourcing Mental Illness) Mental Health in Tech Survey dataset. It aims to raise awareness and assist in early identification of mental health concerns based on workplace and personal factors.

---

## 📌 Project Summary

- **Objective:** Predict whether a person currently has a mental health disorder based on survey responses.
- **Dataset:** OSMI Mental Health in Tech Survey
- **Target Variable:** `Do you currently have a mental health disorder?`
- **Problem Type:** Binary classification (Yes/No)

---

## 🧰 Features Used

- Demographic information (age, gender, region)
- Workplace support (employer attitudes, mental health coverage, anonymity)
- Personal mental health history and awareness
- Work-life balance and stress levels

---

## 🧪 Technologies & Tools

- **Programming Language:**
  - Python
- **Libraries & Frameworks:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost
- **Machine Learning Algorithms:**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - Decision Tree
  - Neural Network (MLPClassifier)
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## ⚙️ Workflow

- **Data Cleaning & Preprocessing:**
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling
  - Feature selection

- **Model Training & Evaluation:**
  - Training various classification models
  - Cross-validation
  - Hyperparameter tuning (where applicable)

- **Model Interpretation:**
  - Feature importance visualization
  - Analysis of influential factors

---

## 📊 Results

- The best-performing model was XGBoost.
- Achieved an accuracy of 76.06%.
- Top predictive factors included:
  - Previous mental health diagnosis
  - Employer support and openness
  - Work interference due to mental health

---

## 🚀 Future Work

- Update the dataset with more recent responses
- Integrate deep learning techniques
- Ensure fairness and reduce prediction bias
---

## 📄 License

- This project is licensed under the [MIT License](LICENSE).

