import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Create model directory
if not os.path.exists('model'):
    os.makedirs('model')

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('Breast_Cancer_Dataset.csv')

# 2. Preprocessing
print("Preprocessing...")
# Drop 'id' and 'Unnamed: 32' if exists
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode Target
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis']) # M -> 1, B -> 0 usually, but let's check
# M = Malignant (1), B = Benign (0) likely. 
# Let's print mapping
print(f"Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler (important for app)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 3. Model Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

print("\nTraining models and calculating metrics...")
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred # Decision Tree has proba too
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    })
    
    print(f"{name} trained. Accuracy: {acc:.4f}")
    
    # Save Model
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

print("\nAll models saved.")

# 4. Save Metrics to CSV for easy reading
results_df = pd.DataFrame(results)
results_df.to_csv('model_metrics.csv', index=False)
print("\nMetrics summary:")
print(results_df)
