"""
Smart Discount Engine
Uses user behavior signals to predict personalized discount tiers.
Features: visits, cart_value, items_viewed, purchase_count, time_on_site (minutes)
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle, os

DISCOUNT_TIERS = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20}

def _generate_training_data():
    np.random.seed(42)
    n = 2000
    visits          = np.random.randint(1, 50, n)
    cart_value      = np.random.uniform(0, 5000, n)
    items_viewed    = np.random.randint(0, 100, n)
    purchase_count  = np.random.randint(0, 20, n)
    time_on_site    = np.random.uniform(0.5, 120, n)

    X = np.column_stack([visits, cart_value, items_viewed, purchase_count, time_on_site])

    # Rule-based labels (ground truth for training)
    score = (
        np.clip(visits / 10, 0, 1) * 0.2 +
        np.clip(cart_value / 2000, 0, 1) * 0.3 +
        np.clip(items_viewed / 50, 0, 1) * 0.15 +
        np.clip(purchase_count / 10, 0, 1) * 0.25 +
        np.clip(time_on_site / 60, 0, 1) * 0.10
    )
    y = np.digitize(score, bins=[0.2, 0.4, 0.6, 0.8]) # 0-4

    return X, y

def train_and_save():
    X, y = _generate_training_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_scaled, y)

    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    with open(model_path, "wb") as f: pickle.dump(model, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    print("✅ Model trained and saved.")
    return model, scaler

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    if not os.path.exists(model_path):
        return train_and_save()
    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)
    return model, scaler

def predict_discount(visits, cart_value, items_viewed, purchase_count, time_on_site):
    model, scaler = load_model()
    features = np.array([[visits, cart_value, items_viewed, purchase_count, time_on_site]])
    features_scaled = scaler.transform(features)
    tier = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0]
    discount_pct = DISCOUNT_TIERS[tier]
    return {
        "tier": tier,
        "discount_percent": discount_pct,
        "confidence": round(float(proba[tier]) * 100, 1),
        "message": _discount_message(discount_pct)
    }

def _discount_message(pct):
    if pct == 0:   return "Welcome! Browse more to unlock exclusive deals."
    if pct == 5:   return "You've earned a 5% loyalty discount!"
    if pct == 10:  return "Great taste! Enjoy 10% off your order."
    if pct == 15:  return "VIP shopper! Here's 15% just for you."
    return "Elite member! Maximum 20% discount unlocked 🎉"
