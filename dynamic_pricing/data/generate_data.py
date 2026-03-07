import pandas as pd
import numpy as np
import os

def generate_retail_data(n_samples=5000, save_path='data/retail_data.csv'):
    np.random.seed(42)
    categories = ['Electronics', 'Clothing', 'Groceries', 'Books', 'Sports', 'Home & Garden']
    data = []

    for i in range(n_samples):
        category = np.random.choice(categories)
        base_prices = {
            'Electronics': np.random.uniform(50, 1500),
            'Clothing': np.random.uniform(10, 200),
            'Groceries': np.random.uniform(1, 50),
            'Books': np.random.uniform(5, 80),
            'Sports': np.random.uniform(15, 500),
            'Home & Garden': np.random.uniform(10, 300)
        }
        base_price = base_prices[category]
        competitor_price_1 = base_price * np.random.uniform(0.8, 1.2)
        competitor_price_2 = base_price * np.random.uniform(0.85, 1.15)
        inventory_level = np.random.randint(0, 500)
        demand_score = np.random.uniform(0, 1)
        day_of_week = np.random.randint(0, 7)
        month = np.random.randint(1, 13)
        is_weekend = 1 if day_of_week >= 5 else 0
        is_holiday_season = 1 if month in [11, 12] else 0
        customer_rating = np.random.uniform(1, 5)
        elasticity = np.random.uniform(-2.5, -0.5)

        if demand_score > 0.7:
            price_multiplier = np.random.uniform(1.05, 1.25)
        elif demand_score < 0.3:
            price_multiplier = np.random.uniform(0.85, 0.99)
        else:
            price_multiplier = np.random.uniform(0.95, 1.10)

        if inventory_level < 20:
            price_multiplier *= 1.1
        elif inventory_level > 400:
            price_multiplier *= 0.95

        min_competitor = min(competitor_price_1, competitor_price_2)
        optimal_price = min(base_price * price_multiplier, min_competitor * 1.05)
        optimal_price = max(optimal_price, base_price * 0.7)
        demand = 100 * (optimal_price / base_price) ** elasticity * demand_score
        revenue = optimal_price * max(demand, 0)

        data.append({
            'product_id': f'P{i:05d}',
            'category': category,
            'base_price': round(base_price, 2),
            'competitor_price_1': round(competitor_price_1, 2),
            'competitor_price_2': round(competitor_price_2, 2),
            'inventory_level': inventory_level,
            'demand_score': round(demand_score, 3),
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'is_holiday_season': is_holiday_season,
            'customer_rating': round(customer_rating, 1),
            'elasticity': round(elasticity, 3),
            'optimal_price': round(optimal_price, 2),
            'expected_revenue': round(revenue, 2)
        })

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {n_samples} samples → {save_path}")
    return df

if __name__ == '__main__':
    df = generate_retail_data()
    print(df.head())