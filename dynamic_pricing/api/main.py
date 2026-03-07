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

@app.get("/")
def root():
    return {"message": "AI Dynamic Pricing Engine is LIVE 🚀"}

@app.post("/predict-price")
def predict_price(req: PricingRequest):
    try:
        category_encoded = encoder.transform([req.category])[0]
    except:
        category_encoded = 0

    is_weekend = 1 if req.day_of_week >= 5 else 0
    is_holiday_season = 1 if req.month in [11, 12] else 0

    features = np.array([[
        req.base_price, req.competitor_price_1, req.competitor_price_2,
        req.inventory_level, req.demand_score, req.day_of_week, req.month,
        is_weekend, is_holiday_season, req.customer_rating, category_encoded
    ]])

    optimal_price = float(model.predict(features)[0])
    price_change_pct = ((optimal_price - req.base_price) / req.base_price) * 100

    if price_change_pct > 5:
        recommendation = f"📈 Raise price by {price_change_pct:.1f}% — High demand detected"
        strategy = "Premium Pricing"
    elif price_change_pct < -5:
        recommendation = f"📉 Lower price by {abs(price_change_pct):.1f}% — Boost sales volume"
        strategy = "Competitive Pricing"
    else:
        recommendation = "✅ Hold current price — Market is balanced"
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