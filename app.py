from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Encoding functions
def encode_gender(gender):
    mapping = {
        'Male': 0,
        'Female': 1,
        'Non-binary/third gender': 2,
        'Prefer to self-describe': 3,
        'Prefer not to say': 4
    }
    return mapping.get(gender, -1)

def encode_self_employed(status):
    mapping = {
        'Yes': 1,
        'No': 0
    }
    return mapping.get(status, -1)

def encode_company_size(size):
    mapping = {
        '1-5': 0,
        '6-25': 1,
        '26-100': 2,
        '100-500': 3,
        '500-1000': 4,
        'More than 1000': 5
    }
    return mapping.get(size, -1)

def encode_tech_company(status):
    mapping = {
        'Yes': 1,
        'No': 0
    }
    return mapping.get(status, -1)

def encode_role_tech(status):
    mapping = {
        'Yes': 1,
        'No': 0
    }
    return mapping.get(status, -1)

def encode_mh_benefits(status):
    mapping = {
        'Yes': 1,
        'No': 0,
        "Don't know": 2
    }
    return mapping.get(status, -1)

def encode_mh_coverage(status):
    mapping = {
        'Yes': 1,
        'No': 0,
        "Don't know": 2
    }
    return mapping.get(status, -1)

def encode_employer_discussion(status):
    mapping = {
        'Yes': 1,
        'No': 0,
        "Don't know": 2
    }
    return mapping.get(status, -1)

def encode_employer_resources(status):
    mapping = {
        'Yes': 1,
        'No': 0,
        "Not sure": 2
    }
    return mapping.get(status, -1)

def encode_anonymity(status):
    mapping = {
        'Yes': 1,
        'No': 0,
        "Don't know": 2
    }
    return mapping.get(status, -1)

def encode_leave_difficulty(level):
    mapping = {
        'Very easy': 0,
        'Somewhat easy': 1,
        'Somewhat difficult': 2,
        'Very difficult': 3,
        "Don't know": 4
    }
    return mapping.get(level, -1)

def encode_country(country):
    mapping = {
        'United States': 0,
        'India': 1,
        'United Kingdom': 2,
        'Canada': 3,
        'Other': 4
    }
    return mapping.get(country, 4)  # Default to 'Other'

def encode_race(race):
    mapping = {
        'White': 0,
        'Black': 1,
        'Asian': 2,
        'Hispanic': 3,
        'Other': 4
    }
    return mapping.get(race, 4)  # Default to 'Other'

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect data from form
            age = int(request.form['age'])
            gender = encode_gender(request.form['gender'])
            self_employed = encode_self_employed(request.form['self_employed'])
            company_size = encode_company_size(request.form['company_size'])
            tech_company = encode_tech_company(request.form['tech_company'])
            role_tech = encode_role_tech(request.form['role_tech'])
            mh_benefits = encode_mh_benefits(request.form['mh_benefits'])
            mh_coverage = encode_mh_coverage(request.form['mh_coverage'])
            employer_discussion = encode_employer_discussion(request.form['employer_discussion'])
            employer_resources = encode_employer_resources(request.form['employer_resources'])
            anonymity = encode_anonymity(request.form['anonymity'])
            leave_difficulty = encode_leave_difficulty(request.form['leave_difficulty'])
            

            # Prepare feature array
            data = np.array([[age, gender, self_employed, company_size, tech_company,
                              role_tech, mh_benefits, mh_coverage, employer_discussion,
                              employer_resources, anonymity, leave_difficulty,
                              ]])

            # Prediction
            prediction = model.predict(data)

            # Interpretation
            result = 'At Risk of Depression' if prediction[0] == 1 else 'Not at Risk of Depression'

            return render_template('result.html', prediction=result)

        except Exception as e:
            return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
