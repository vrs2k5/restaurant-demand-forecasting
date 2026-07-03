from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    restaurant_id = db.Column(db.Integer)
    menu_item_name = db.Column(db.String(100))
    meal_type = db.Column(db.String(20))
    weather_condition = db.Column(db.String(20))
    has_promotion = db.Column(db.Boolean)
    special_event = db.Column(db.Boolean)

    linear_prediction = db.Column(db.Float)
    random_forest_prediction = db.Column(db.Float)
    xgboost_prediction = db.Column(db.Float)
    lightgbm_prediction = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
