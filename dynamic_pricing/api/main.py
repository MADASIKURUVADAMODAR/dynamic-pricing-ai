from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import json

app = FastAPI(title="AI Dynamic Pricing Engine", version="1.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model_data = joblib.load('model/pricing_model.pkl')
model = model_data['model']
encoder = model_data['encoder']
target_mode = model_data.get('target_mode', 'absolute_price')
model_features = model_data.get('features', [
    'base_price', 'competitor_price_1', 'competitor_price_2',
    'inventory_level', 'demand_score', 'day_of_week', 'month',
    'is_weekend', 'is_holiday_season', 'customer_rating', 'category_encoded'
])

class PricingRequest(BaseModel):
    category: str
    base_price: float
    competitor_price_1: float
    competitor_price_2: float
    inventory_level: int
    demand_score: float
    day_of_week: int
    month: int
    customer_rating: float


def build_feature_map(req: PricingRequest, category_encoded: int) -> dict:
    is_weekend = 1 if req.day_of_week >= 5 else 0
    is_holiday_season = 1 if req.month in [11, 12] else 0
    avg_comp_price = (req.competitor_price_1 + req.competitor_price_2) / 2.0
    safe_base_price = max(req.base_price, 1e-6)

    price_gap_1 = req.base_price - req.competitor_price_1
    price_gap_2 = req.base_price - req.competitor_price_2
    inventory_pressure = req.demand_score / (req.inventory_level + 1.0)
    market_pressure = avg_comp_price / safe_base_price
    seasonal_weight = 1.0 + (0.25 * is_holiday_season) + (0.10 * is_weekend) + (0.05 if req.month in [6, 7, 8] else 0.0)
    demand_inventory_ratio = (req.demand_score * seasonal_weight) / (req.inventory_level + 1.0)

    return {
        'base_price': req.base_price,
        'competitor_price_1': req.competitor_price_1,
        'competitor_price_2': req.competitor_price_2,
        'inventory_level': req.inventory_level,
        'demand_score': req.demand_score,
        'day_of_week': req.day_of_week,
        'month': req.month,
        'is_weekend': is_weekend,
        'is_holiday_season': is_holiday_season,
        'customer_rating': req.customer_rating,
        'category_encoded': category_encoded,
        'price_gap_1': price_gap_1,
        'price_gap_2': price_gap_2,
        'inventory_pressure': inventory_pressure,
        'market_pressure': market_pressure,
        'seasonal_weight': seasonal_weight,
        'demand_inventory_ratio': demand_inventory_ratio,
    }

@app.get("/")
def root():
    return {"message": "AI Dynamic Pricing Engine is LIVE 🚀"}

@app.post("/predict-price")
def predict_price(req: PricingRequest):
    try:
        category_encoded = encoder.transform([req.category])[0]
    except:
        category_encoded = 0

    feature_map = build_feature_map(req, category_encoded)
    features = np.array([[feature_map[col] for col in model_features]])

    raw_prediction = float(model.predict(features)[0])
    if target_mode == 'relative_delta':
        optimal_price = req.base_price * (1.0 + raw_prediction)
    else:
        optimal_price = raw_prediction
    price_change_pct = ((optimal_price - req.base_price) / req.base_price) * 100

    avg_competitor_price = (req.competitor_price_1 + req.competitor_price_2) / 2.0
    high_demand_low_inventory = req.demand_score >= 0.7 and req.inventory_level <= 50
    competitors_cheaper = avg_competitor_price < req.base_price * 0.97
    holiday_season = req.month in [11, 12]

    if high_demand_low_inventory:
        recommendation = f"📈 Raise pricing due to strong demand and limited inventory ({price_change_pct:+.1f}%)"
        strategy = "Scarcity Pricing"
    elif competitors_cheaper:
        recommendation = f"📉 Reduce pricing for competitiveness ({price_change_pct:+.1f}%)"
        strategy = "Competitive Pricing"
    elif holiday_season:
        recommendation = f"🎄 Increase pricing due to seasonal surge ({price_change_pct:+.1f}%)"
        strategy = "Seasonal Premium"
    else:
        recommendation = f"✅ Maintain stable pricing ({price_change_pct:+.1f}%)"
        strategy = "Stable Pricing"

    return {
        "optimal_price": round(optimal_price, 2),
        "price_change_pct": round(price_change_pct, 2),
        "recommendation": recommendation,
        "strategy": strategy
    }

@app.get("/feature-importance")
def get_feature_importance():
    with open('model/feature_importance.json') as f:
        return json.load(f)

@app.get("/categories")
def get_categories():
    return {"categories": list(encoder.classes_)}