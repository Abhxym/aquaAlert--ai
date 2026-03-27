import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import joblib
import os

train_path = r"C:/Users/Admin/Downloads/playground-series-s5e3/train.csv"
print("Loading Kaggle Dataset from", train_path)
df = pd.read_csv(train_path)

X_df = df.drop(columns=['id', 'day', 'rainfall'])
y = df['rainfall']

print("Compiling Data...")
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

xgb_kaggle = XGBClassifier(n_estimators=450, learning_rate=0.03, max_depth=7, subsample=0.85, colsample_bytree=0.85, use_label_encoder=False, eval_metric='logloss')

print("Training XGBoost on 10,000+ Atmospheric Arrays...")
xgb_kaggle.fit(X_train_scaled, y_train)

models_dir = os.path.join("models", "saved")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(xgb_kaggle, os.path.join(models_dir, "kaggle_rainfall_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "kaggle_scaler.pkl"))

print("Kaggle Models Successfully Written to models/saved!")
