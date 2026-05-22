import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import json


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Build derived business signals used by both training and inference.
    df = df.copy()

    avg_competitor_price = (df['competitor_price_1'] + df['competitor_price_2']) / 2.0
    safe_base_price = np.clip(df['base_price'], 1e-6, None)

    df['price_gap_1'] = df['base_price'] - df['competitor_price_1']
    df['price_gap_2'] = df['base_price'] - df['competitor_price_2']
    df['inventory_pressure'] = df['demand_score'] / (df['inventory_level'] + 1.0)
    df['market_pressure'] = avg_competitor_price / safe_base_price

    df['seasonal_weight'] = (
        1.0
        + 0.25 * df['is_holiday_season']
        + 0.10 * df['is_weekend']
        + 0.05 * df['month'].isin([6, 7, 8]).astype(float)
    )
    df['demand_inventory_ratio'] = (df['demand_score'] * df['seasonal_weight']) / (df['inventory_level'] + 1.0)

    numeric_cols = [
        'price_gap_1', 'price_gap_2', 'inventory_pressure',
        'market_pressure', 'seasonal_weight', 'demand_inventory_ratio'
    ]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def train_pricing_model(data_path='data/retail_data.csv', model_path='model/pricing_model.pkl'):
    print("📊 Loading data...")
    df = pd.read_csv(data_path)

    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Preserve existing behavioral features then enrich with business-driven signals.
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    df = add_engineered_features(df)

    feature_cols = [
        'base_price', 'competitor_price_1', 'competitor_price_2',
        'inventory_level', 'demand_score', 'day_of_week', 'month',
        'is_weekend', 'is_holiday_season', 'customer_rating', 'category_encoded',
        'price_gap_1', 'price_gap_2', 'inventory_pressure',
        'market_pressure', 'seasonal_weight', 'demand_inventory_ratio'
    ]

    X = df[feature_cols]
    y_price = df['optimal_price']
    y_delta = (df['optimal_price'] - df['base_price']) / np.clip(df['base_price'], 1e-6, None)

    X_train, X_test, y_train_delta, y_test_delta, y_train_price, y_test_price = train_test_split(
        X, y_delta, y_price, test_size=0.2, random_state=42
    )

    print("🤖 Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train_delta)

    y_pred_delta = model.predict(X_test)
    y_pred_price = X_test['base_price'].to_numpy() * (1.0 + y_pred_delta)
    mae = mean_absolute_error(y_test_price, y_pred_price)
    r2 = r2_score(y_test_price, y_pred_price)

    print("\n📈 Model Evaluation")
    print(f"✅ MAE: ${mae:.2f}")
    print(f"✅ R² Score: {r2:.4f}")

    os.makedirs('model', exist_ok=True)
    joblib.dump(
        {
            'model': model,
            'encoder': le,
            'features': feature_cols,
            'target_mode': 'relative_delta'
        },
        model_path
    )

    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance_raw': model.feature_importances_
    }).sort_values('importance_raw', ascending=False)
    total_importance = feature_importance_df['importance_raw'].sum()
    if total_importance > 0:
        feature_importance_df['importance_pct'] = (feature_importance_df['importance_raw'] / total_importance) * 100.0
    else:
        feature_importance_df['importance_pct'] = 0.0

    print("\n🔍 Feature Importance Ranking")
    for rank, row in enumerate(feature_importance_df.itertuples(index=False), start=1):
        print(f"{rank:>2}. {row.feature:<24} {row.importance_pct:>6.2f}%")

    # Keep endpoint/dashboard compatibility: JSON stays a flat numeric mapping.
    importance = {
        row.feature: round(float(row.importance_pct), 4)
        for row in feature_importance_df.itertuples(index=False)
    }
    with open('model/feature_importance.json', 'w') as f:
        json.dump(importance, f, indent=2)

    print(f"💾 Model saved → {model_path}")
    return model, le, {'mae': mae, 'r2': r2}

if __name__ == '__main__':
    train_pricing_model()