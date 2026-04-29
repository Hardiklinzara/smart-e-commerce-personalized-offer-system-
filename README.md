# VÊTIR — Smart Fashion E-Commerce
### Python + ML Discount Engine

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open in browser
# http://localhost:5000
```

---

## 🤖 How the ML Discount Engine Works

The discount engine uses a **Gradient Boosting Classifier** trained on 2,000
synthetic user behavior samples. It predicts one of 5 discount tiers in real-time.

### Input Features (signals collected per session):

| Feature | Description |
|---|---|
| `visits` | Number of times user has visited the site |
| `cart_value` | Current cart total (₹) |
| `items_viewed` | How many product pages opened |
| `purchase_count` | Past completed orders |
| `time_on_site` | Minutes spent in current session |

### Discount Tiers:

| Tier | Discount | Who gets it |
|---|---|---|
| 0 | 0% | New visitor, empty cart |
| 1 | 5% | Returning visitor, some browsing |
| 2 | 10% | Active shopper, decent cart value |
| 3 | 15% | High engagement, past purchases |
| 4 | 20% | Loyal, high-value returning customer |

---

## 📁 Project Structure

```
smart_fashion/
├── app.py                  # Flask backend + REST API
├── requirements.txt
├── ml/
│   └── discount_engine.py  # ML model (GradientBoosting)
├── templates/
│   └── index.html          # Full frontend (HTML/CSS/JS)
└── data/
    └── shop.db             # SQLite database (auto-created)
```

## 🛠 API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/api/products` | List products (filter by `?category=`) |
| GET | `/api/cart` | Get current cart |
| POST | `/api/cart/add` | Add item `{product_id}` |
| POST | `/api/cart/remove` | Remove item `{product_id}` |
| POST | `/api/view_product` | Track product view |
| GET | `/api/discount` | Get ML-predicted discount |
| POST | `/api/checkout` | Place order |
| GET | `/api/stats` | Admin stats |
