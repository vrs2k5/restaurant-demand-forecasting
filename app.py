from flask import Flask, request, render_template, jsonify
from datetime import datetime
import logging

from config import Config
from models import db, Prediction
from model import DemandForecaster

import os

app = Flask(__name__)
app.config.from_object(Config)

# Ensure database directory exists
db_path = os.path.join(app.root_path, "database")
os.makedirs(db_path, exist_ok=True)

db.init_app(app)

with app.app_context():
    db.create_all()

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

forecaster = DemandForecaster("dataset/restaurant_sales_data.csv")
forecaster.load()
forecaster.load_data()  # populates forecaster.df for dashboard/analytics routes


# -----------------------------
# Home / Predict
# -----------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        restaurants=forecaster.restaurants,
        restaurant_item_map=forecaster.restaurant_item_map,
        weather_options=forecaster.weather_options,
        meal_types=forecaster.meal_types,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        date = request.form["date"]
        restaurant_id = int(request.form["restaurant_id"])
        item = request.form["item"]
        meal_type = request.form["meal_type"]
        weather_condition = request.form["weather_condition"]
        has_promotion = request.form.get("has_promotion") == "on"
        special_event = request.form.get("special_event") == "on"

        preds = forecaster.predict(
            date, restaurant_id, item, meal_type, weather_condition,
            has_promotion, special_event,
        )

        record = Prediction(
            date=date,
            restaurant_id=restaurant_id,
            menu_item_name=item,
            meal_type=meal_type,
            weather_condition=weather_condition,
            has_promotion=has_promotion,
            special_event=special_event,
            linear_prediction=preds["linear"],
            random_forest_prediction=preds["random_forest"],
            xgboost_prediction=preds["xgboost"],
            lightgbm_prediction=preds["lightgbm"],
        )
        db.session.add(record)
        db.session.commit()

        logging.info(f"Prediction | Restaurant={restaurant_id} Item={item} Date={date}")

        return render_template(
            "result.html",
            date=date,
            restaurant_id=restaurant_id,
            item=item,
            meal_type=meal_type,
            weather_condition=weather_condition,
            preds=preds,
            best_model=forecaster.best_model_name,
        )

    except Exception as e:
        logging.error(str(e))
        return render_template("error.html", error=str(e))


# -----------------------------
# Inventory Recommendations
# -----------------------------

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return render_template(
            "recommend.html",
            restaurants=forecaster.restaurants,
            weather_options=forecaster.weather_options,
            recommendations=None,
        )

    try:
        date = request.form["date"]
        restaurant_id = int(request.form["restaurant_id"])
        weather_condition = request.form["weather_condition"]
        has_promotion = request.form.get("has_promotion") == "on"
        special_event = request.form.get("special_event") == "on"

        recommendations = forecaster.recommend_for_restaurant(
            date, restaurant_id, weather_condition, has_promotion, special_event
        )

        return render_template(
            "recommend.html",
            restaurants=forecaster.restaurants,
            weather_options=forecaster.weather_options,
            recommendations=recommendations,
            selected_restaurant=restaurant_id,
            date=date,
            best_model=forecaster.best_model_name,
        )
    except Exception as e:
        logging.error(str(e))
        return render_template("error.html", error=str(e))


# -----------------------------
# Prediction History
# -----------------------------

@app.route("/history")
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()
    return render_template("history.html", predictions=predictions)


# -----------------------------
# Dashboard
# -----------------------------

@app.route("/dashboard")
def dashboard():
    total_predictions = Prediction.query.count()
    latest = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()

    df = forecaster.df
    total_records = len(df)
    total_restaurants = df["restaurant_id"].nunique()
    total_items = df["menu_item_name"].nunique()
    avg_quantity = round(df["quantity_sold"].mean(), 2)
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = df["date"].max().strftime("%Y-%m-%d")

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        latest=latest,
        total_records=total_records,
        total_restaurants=total_restaurants,
        total_items=total_items,
        avg_quantity=avg_quantity,
        date_min=date_min,
        date_max=date_max,
    )


# -----------------------------
# Analytics
# -----------------------------

@app.route("/analytics")
def analytics():
    try:
        df = forecaster.df.copy()
        df["month"] = df["date"].dt.strftime("%Y-%m")

        monthly = df.groupby("month")["quantity_sold"].sum().reset_index()
        item_sales = (
            df.groupby("menu_item_name")["quantity_sold"]
            .sum()
            .sort_values(ascending=False)
        )
        weather_sales = df.groupby("weather_condition")["quantity_sold"].mean().reset_index()
        promo_effect = df.groupby("has_promotion")["quantity_sold"].mean().reset_index()

        return render_template(
            "analytics.html",
            months=monthly["month"].tolist(),
            monthly=monthly["quantity_sold"].tolist(),
            items=item_sales.index.tolist(),
            item_sales=item_sales.values.tolist(),
            weather_labels=weather_sales["weather_condition"].tolist(),
            weather_values=[round(v, 1) for v in weather_sales["quantity_sold"].tolist()],
            promo_labels=["No Promotion", "With Promotion"],
            promo_values=[round(v, 1) for v in promo_effect["quantity_sold"].tolist()],
        )
    except Exception as e:
        logging.error(str(e))
        return render_template("error.html", error=str(e))


# -----------------------------
# Model Metrics / Comparison
# -----------------------------

@app.route("/metrics")
def metrics():
    return render_template(
        "metrics.html",
        metrics=forecaster.metrics,
        best_model=forecaster.best_model_name,
        feature_importance=forecaster.feature_importance,
    )


# -----------------------------
# REST API
# -----------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        preds = forecaster.predict(
            data["date"],
            int(data["restaurant_id"]),
            data["item"],
            data["meal_type"],
            data["weather_condition"],
            bool(data.get("has_promotion", False)),
            bool(data.get("special_event", False)),
        )
        return jsonify(preds)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/recommend/<int:restaurant_id>")
def api_recommend(restaurant_id):
    try:
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        weather_condition = request.args.get("weather", "Sunny")
        recs = forecaster.recommend_for_restaurant(date, restaurant_id, weather_condition)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/dataset")
def dataset_info():
    df = forecaster.df
    return jsonify({
        "records": len(df),
        "restaurants": int(df["restaurant_id"].nunique()),
        "menu_items": int(df["menu_item_name"].nunique()),
        "average_quantity_sold": round(float(df["quantity_sold"].mean()), 2),
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(forecaster.metrics)


@app.route("/api/history")
def api_history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()
    result = [{
        "id": p.id,
        "date": p.date,
        "restaurant_id": p.restaurant_id,
        "menu_item_name": p.menu_item_name,
        "lightgbm_prediction": p.lightgbm_prediction,
        "xgboost_prediction": p.xgboost_prediction,
        "created_at": str(p.created_at),
    } for p in predictions]
    return jsonify(result)


# -----------------------------
# Health Check
# -----------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "Running",
        "models": list(forecaster.models.keys()),
        "best_model": forecaster.best_model_name,
        "database": "Connected",
    })


# -----------------------------
# Error Handlers
# -----------------------------

@app.errorhandler(404)
def page_not_found(error):
    return render_template("error.html", error="404 - Page Not Found"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("error.html", error="500 - Internal Server Error"), 500


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)