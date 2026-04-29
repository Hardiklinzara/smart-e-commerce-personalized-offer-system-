"""
Smart Fashion E-Commerce Backend
Flask + SQLite + ML Discount Engine
"""

from flask import Flask, render_template, jsonify, request, session
import sqlite3, json, os, sys, uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ml.discount_engine import predict_discount, train_and_save

app = Flask(__name__)
app.secret_key = "smart_fashion_secret_2024"
DB = os.path.join(os.path.dirname(__file__), "data", "shop.db")

# ── Products catalogue ────────────────────────────────────────────────────────
PRODUCTS = [
    {"id":1,"name":"Noir Silk Blouse","category":"Tops","price":2499,"original":3200,"image":"https://images.unsplash.com/photo-1598554747436-c9293d6a588f?w=600&q=80","rating":4.8,"reviews":124,"tag":"Bestseller"},
    {"id":2,"name":"Linen Wide-Leg Trousers","category":"Bottoms","price":3199,"original":3999,"image":"https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=600&q=80","rating":4.6,"reviews":89,"tag":"New"},
    {"id":3,"name":"Cashmere Turtleneck","category":"Tops","price":5499,"original":6800,"image":"https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=600&q=80","rating":4.9,"reviews":203,"tag":"Premium"},
    {"id":4,"name":"Pleated Midi Skirt","category":"Bottoms","price":2199,"original":2800,"image":"https://images.unsplash.com/photo-1583496661160-fb5218ees0b7?w=600&q=80","rating":4.5,"reviews":67,"tag":"Trending"},
    {"id":5,"name":"Structured Blazer","category":"Outerwear","price":6999,"original":8500,"image":"https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=600&q=80","rating":4.7,"reviews":156,"tag":"Classic"},
    {"id":6,"name":"Wrap Sundress","category":"Dresses","price":2899,"original":3500,"image":"https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=600&q=80","rating":4.6,"reviews":98,"tag":"Summer"},
    {"id":7,"name":"Ribbed Knit Cardigan","category":"Tops","price":3599,"original":4200,"image":"https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=600&q=80","rating":4.8,"reviews":175,"tag":"Cozy"},
    {"id":8,"name":"High-Rise Straight Jeans","category":"Bottoms","price":3999,"original":4800,"image":"https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=600&q=80","rating":4.5,"reviews":312,"tag":"Staple"},
]

# ── DB init ───────────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            session_id TEXT PRIMARY KEY,
            visits INTEGER DEFAULT 1,
            items_viewed INTEGER DEFAULT 0,
            purchase_count INTEGER DEFAULT 0,
            total_spent REAL DEFAULT 0,
            first_seen TEXT,
            last_seen TEXT
        );
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            product_id INTEGER,
            quantity INTEGER DEFAULT 1,
            added_at TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            items TEXT,
            subtotal REAL,
            discount_pct INTEGER,
            final_total REAL,
            placed_at TEXT
        );
    """)
    con.commit(); con.close()

def get_or_create_user(sid):
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE session_id=?", (sid,))
    row = cur.fetchone()
    now = datetime.utcnow().isoformat()
    if not row:
        cur.execute("INSERT INTO users VALUES (?,1,0,0,0,?,?)", (sid, now, now))
        con.commit()
    else:
        cur.execute("UPDATE users SET visits=visits+1, last_seen=? WHERE session_id=?", (now, sid))
        con.commit()
    cur.execute("SELECT * FROM users WHERE session_id=?", (sid,))
    row = cur.fetchone()
    con.close()
    return {"session_id":row[0],"visits":row[1],"items_viewed":row[2],
            "purchase_count":row[3],"total_spent":row[4]}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    session["start_time"] = session.get("start_time", datetime.utcnow().timestamp())
    get_or_create_user(session["sid"])
    return render_template("index.html")

@app.route("/api/products")
def api_products():
    category = request.args.get("category", "all")
    if category == "all":
        return jsonify(PRODUCTS)
    return jsonify([p for p in PRODUCTS if p["category"] == category])

@app.route("/api/cart", methods=["GET"])
def get_cart():
    sid = session.get("sid")
    if not sid: return jsonify([])
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT product_id, quantity FROM cart WHERE session_id=?", (sid,))
    rows = cur.fetchall(); con.close()
    cart = []
    for pid, qty in rows:
        prod = next((p for p in PRODUCTS if p["id"]==pid), None)
        if prod: cart.append({**prod, "quantity": qty})
    return jsonify(cart)

@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    sid = session.get("sid")
    data = request.json
    pid = data.get("product_id")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT id,quantity FROM cart WHERE session_id=? AND product_id=?", (sid,pid))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE cart SET quantity=quantity+1 WHERE id=?", (row[0],))
    else:
        cur.execute("INSERT INTO cart (session_id,product_id,quantity,added_at) VALUES (?,?,1,?)",
                    (sid, pid, datetime.utcnow().isoformat()))
    con.commit(); con.close()
    return jsonify({"status":"ok"})

@app.route("/api/cart/remove", methods=["POST"])
def remove_from_cart():
    sid = session.get("sid")
    pid = request.json.get("product_id")
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("DELETE FROM cart WHERE session_id=? AND product_id=?", (sid,pid))
    con.commit(); con.close()
    return jsonify({"status":"ok"})

@app.route("/api/view_product", methods=["POST"])
def view_product():
    sid = session.get("sid")
    if sid:
        con = sqlite3.connect(DB)
        con.execute("UPDATE users SET items_viewed=items_viewed+1 WHERE session_id=?", (sid,))
        con.commit(); con.close()
    return jsonify({"status":"ok"})

@app.route("/api/discount")
def get_discount():
    sid = session.get("sid")
    if not sid: return jsonify({"discount_percent":0,"message":"","tier":0,"confidence":0})
    user = get_or_create_user(sid)
    start = session.get("start_time", datetime.utcnow().timestamp())
    time_on_site = (datetime.utcnow().timestamp() - start) / 60

    # Cart value
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT product_id, quantity FROM cart WHERE session_id=?", (sid,))
    rows = cur.fetchall(); con.close()
    cart_value = sum(
        next((p["price"] for p in PRODUCTS if p["id"]==pid), 0) * qty
        for pid, qty in rows
    )
    result = predict_discount(
        visits=user["visits"],
        cart_value=cart_value,
        items_viewed=user["items_viewed"],
        purchase_count=user["purchase_count"],
        time_on_site=max(time_on_site, 0.1)
    )
    return jsonify(result)

@app.route("/api/checkout", methods=["POST"])
def checkout():
    sid = session.get("sid")
    data = request.json
    discount_pct = data.get("discount_pct", 0)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT product_id, quantity FROM cart WHERE session_id=?", (sid,))
    rows = cur.fetchall()
    if not rows: con.close(); return jsonify({"error":"Cart empty"}), 400
    items, subtotal = [], 0
    for pid, qty in rows:
        prod = next((p for p in PRODUCTS if p["id"]==pid), None)
        if prod:
            items.append({"name":prod["name"],"qty":qty,"price":prod["price"]})
            subtotal += prod["price"] * qty
    final = subtotal * (1 - discount_pct/100)
    cur.execute("INSERT INTO orders (session_id,items,subtotal,discount_pct,final_total,placed_at) VALUES (?,?,?,?,?,?)",
                (sid, json.dumps(items), subtotal, discount_pct, final, datetime.utcnow().isoformat()))
    cur.execute("DELETE FROM cart WHERE session_id=?", (sid,))
    cur.execute("UPDATE users SET purchase_count=purchase_count+1, total_spent=total_spent+? WHERE session_id=?", (final, sid))
    con.commit(); con.close()
    return jsonify({"status":"success","subtotal":subtotal,"discount_pct":discount_pct,"final":final})

@app.route("/api/stats")
def stats():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM orders")
    orders = cur.fetchone()[0]
    cur.execute("SELECT COALESCE(SUM(final_total),0) FROM orders")
    revenue = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT session_id) FROM users")
    users = cur.fetchone()[0]
    con.close()
    return jsonify({"orders":orders,"revenue":round(revenue,2),"users":users})

if __name__ == "__main__":
    init_db()
    print("🤖 Training ML discount model...")
    train_and_save()
    print("🚀 Starting Smart Fashion server at http://localhost:5000")
    app.run(debug=True, port=5000)
