# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

def train_model(data_path='data/corrosion_data.csv', model_dir='models'):
    """
    Loads data, preprocesses it, trains a RandomForestRegressor model,
    evaluates its performance, and saves the final model and scaler artifacts.
    """
    print("Starting model training process...")
    df = pd.read_csv(data_path)

    # 1. Define Features (X) and Target (y)
    features = ['Temperature_C', 'pH', 'Chloride_ppm', 'Flow_Velocity_ms']
    target = 'Corrosion_Rate_mm_per_year'
    X = df[features]
    y = df[target]

    # 2. Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the model
    # A RandomForest is a great choice as it's powerful and handles complex relationships well.
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate the model's performance on unseen test data
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    print(f"\nModel training complete. R² Score on Test Data: {score:.4f}")

    # 6. Save the trained model and the scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'corrosion_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print(f"Model and scaler have been saved to the '{model_dir}' directory.")

if __name__ == "__main__":
    train_model()