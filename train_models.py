import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# -------------------------
# 1️⃣ Load and Clean Data
# -------------------------
DATA_PATH = "D:/DM PROJECT/OSMI 2019 Mental Health in Tech Survey Results - OSMI Mental Health in Tech Survey 2019.csv"
TARGET_COLUMN = "Do you *currently* have a mental health disorder?"

# Load dataset
df = pd.read_csv(DATA_PATH)

# Ensure target variable exists
if TARGET_COLUMN not in df.columns:
    raise KeyError(f"❌ ERROR: Target variable '{TARGET_COLUMN}' not found in dataset!")

# Drop columns with too many missing values (threshold > 50%)
df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)

# Fill missing values (forward fill, then backward fill for stability)
df_cleaned.fillna(method='ffill', inplace=True)
df_cleaned.fillna(method='bfill', inplace=True)

# -------------------------
# 2️⃣ Encode Categorical Data
# -------------------------
le = LabelEncoder()
for col in df_cleaned.select_dtypes(include=['object', 'bool']).columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))

# -------------------------
# 3️⃣ Feature Selection Using Mutual Information
# -------------------------
X = df_cleaned.drop(columns=[TARGET_COLUMN])
y = df_cleaned[TARGET_COLUMN]

mi_scores = mutual_info_classif(X, y, random_state=42)
selected_features = X.columns[mi_scores > np.mean(mi_scores)]  # Select features above mean MI score

# Ensure at least some features are selected
if selected_features.empty:
    selected_features = X.columns  # If MI selection fails, keep all features

X = X[selected_features]

# -------------------------
# 4️⃣ Train-Test Split & Scaling
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# 5️⃣ Define Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# -------------------------
# 6️⃣ Train & Evaluate Models
# -------------------------
results = []
accuracies = {}
conf_matrices = {}
reports = {}
trained_models = {}

for name, model in models.items():
    print(f"🔹 Training {name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    
    # Confusion Matrix
    conf_matrices[name] = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON storage
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)  # Convert to dict for JSON storage
    reports[name] = report
    
    # Append results
    results.append(f"{name} Accuracy: {accuracy:.4f}\n{classification_report(y_test, y_pred)}\n")
    
    # Save trained model
    trained_models[name] = model

# -------------------------
# 7️⃣ Save Results
# -------------------------
with open("initial.txt", "w") as f:
    f.writelines(results)

with open("results.json", "w") as json_file:
    json.dump({
        "accuracies": accuracies,
        "conf_matrices": conf_matrices,
        "reports": reports,
        "feature_names": selected_features.tolist()  # Store feature names for explainability
    }, json_file, indent=4)

# Save trained models
with open("trained_model.pkl", "wb") as f:
    pickle.dump(trained_models, f)

# Save scaler for future use
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Training Complete! Files saved:")
print("📂 initial.txt -> Contains model accuracy reports")
print("📂 results.json -> Contains structured accuracy, confusion matrix, and reports")
print("📂 trained_model.pkl -> Contains trained ML models")
print("📂 scaler.pkl -> StandardScaler used for feature scaling")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
