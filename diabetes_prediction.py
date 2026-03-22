
# ============================================================
#   DIABETES PREDICTION
#   Model: Logistic Regression
#   Dataset: Pima Indians Diabetes (Kaggle)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── 1. LOAD DATASET ─────────────────────────────────────────
df = pd.read_csv("project_2/diabetes.csv")
 
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution:")
print(df["Outcome"].value_counts())
# Outcome: 1 = Diabetic, 0 = Not Diabetic
 
# ── 2. HANDLE MISSING VALUES ────────────────────────────────
# Glucose, BMI etc. can't be 0 in real life — replace with median
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())
 
print("\nZero values replaced with median!")
 
# ── 3. SPLIT FEATURES AND LABEL ─────────────────────────────
X = df.drop("Outcome", axis=1)   # all columns except label
y = df["Outcome"]                 # label column
 
# ── 4. TRAIN-TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 5. FEATURE SCALING ──────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
X_test_scaled  = scaler.transform(X_test)        # sirf transform on test
 
# ── 6. TRAIN MODEL ──────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
 
print("\nModel training complete!")
 
# ── 7. EVALUATE ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("  [TN  FP]")
print("  [FN  TP]")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))

# ── 8. FEATURE IMPORTANCE ───────────────────────────────────
# Logistic Regression gives weights to each feature
# Higher weight = more important feature
print("\nFeature Weights (importance):")
feature_names = X.columns
weights = model.coef_[0]
for name, weight in sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:20s} : {weight:.4f}")