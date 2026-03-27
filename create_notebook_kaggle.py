import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

def add_markdown(text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(code):
    nb.cells.append(nbf.v4.new_code_cell(code))

add_markdown("# Kaggle S5E3: Advanced Meteorological Rainfall Predictor")
add_markdown("This notebook strictly maps raw atmospheric sensors (Dewpoint, Cloud cover, Sun, Wind) natively to true rainfall probabilities using the massive official `playground-series-s5e3` Kaggle Dataset!")

add_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Ensure clean plotting inside VS Code natively
plt.style.use('dark_background')

# Seamlessly binding directly to the local User downloads folder exactly where the dataset resides!
train_path = r"C:/Users/Admin/Downloads/playground-series-s5e3/train.csv"
df = pd.read_csv(train_path)

print(f"✅ Kaggle Dataset Loaded! Total Atmospheric Records: {df.shape[0]:,}")
df.head()
""")

add_markdown("## Intelligent Feature Exploration")
add_code("""
# Drop irrelevant categorical/ID timelines
X_df = df.drop(columns=['id', 'day', 'rainfall'])
y = df['rainfall']

# Plot correlation bounds to visualize atmospheric triggers!
plt.figure(figsize=(10, 8))
sns.heatmap(X_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Atmospheric Parameter Vector Analysis", weight='bold', fontsize=14)
plt.tight_layout()
plt.show()
""")

add_markdown("## Extreme Gradient Boosting (XGBoost) Architecture Engine")
add_code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Stratify the matrix to ensure balanced rainfall nodes
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Booting the XGBoost Subsystem
xgb_kaggle = XGBClassifier(
    n_estimators=450,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("🚀 Compiling Massive Kaggle Matrix, Building Trees...")
xgb_kaggle.fit(X_train_scaled, y_train)

# Inference Verification
preds = xgb_kaggle.predict(X_test_scaled)
acc = accuracy_score(y_test, preds) * 100

print("\\n==================================")
print(f"✅ KAGGLE ENGINE VALIDATED: {acc:.2f}% ACCURACY!")
print("==================================\\n")
print(classification_report(y_test, preds))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, preds), annot=True, cmap='Blues', fmt='d')
plt.title('Kaggle Validation Output')
plt.xlabel('Predicted')
plt.ylabel('Actual Truth')
plt.show()
""")

add_markdown("## Serialize Application Matrix")
add_markdown("We are physically converting the exact XGBoost Model weights and the explicit `MinMaxScaler()` math formulas out of this computational notebook into raw `.pkl` objects, allowing the React UI and FastAPI backend to invoke them seamlessly within 10 milliseconds.")

add_code("""
# Automatically serialize models for backend route architecture
models_dir = os.path.join("models", "saved")
if not os.path.exists("models"):
    models_dir = os.path.join("..", "models", "saved")
os.makedirs(models_dir, exist_ok=True)

# Save Both the Neural Logic Node AND the Math Scaler Parameter 
joblib.dump(xgb_kaggle, os.path.join(models_dir, "kaggle_rainfall_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "kaggle_scaler.pkl"))

print(f"📦 Advanced Kaggle Matrix explicitly compressed into: {models_dir}")
""")

if not os.path.exists("notebooks"):
    os.makedirs("notebooks", exist_ok=True)

with open("notebooks/kaggle_s5e3_predictor.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
    
print("Kaggle Python Notebook Generated.")
