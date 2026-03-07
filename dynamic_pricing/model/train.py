import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json

def train_pricing_model(data_path='data/retail_data.csv', model_path='model/pricing_model.pkl'):
    print("📊 Loading data...")
    df = pd.read_csv(data_path)

    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    feature_cols = [
        'base_price', 'competitor_price_1', 'competitor_price_2',
        'inventory_level', 'demand_score', 'day_of_week', 'month',
        'is_weekend', 'is_holiday_season', 'customer_rating', 'category_encoded'
    ]

    X = df[feature_cols]
    y = df['optimal_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 Training Gradient Boosting model...")
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ MAE: ${mae:.2f}  |  R² Score: {r2:.4f}")

    os.makedirs('model', exist_ok=True)
    joblib.dump({'model': model, 'encoder': le, 'features': feature_cols}, model_path)

    importance = dict(zip(feature_cols, model.feature_importances_))
    with open('model/feature_importance.json', 'w') as f:
        json.dump(importance, f, indent=2)

    print(f"💾 Model saved → {model_path}")
    return model, le, {'mae': mae, 'r2': r2}

if __name__ == '__main__':
    train_pricing_model()