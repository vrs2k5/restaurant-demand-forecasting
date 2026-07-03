"""
Core demand forecasting engine.

Builds per-(restaurant, menu item) sales history features, trains and
compares four regression models, and serves predictions + next-day
inventory recommendations.

NOTE ON DATA: this project uses a synthetic restaurant sales dataset
(50 fictional restaurants, 14 menu items, Jan 2024 - Jan 2025). It is
NOT scraped real-world operational data. That's stated plainly here and
in the README so the modeling choices below can be understood honestly.

NOTE ON LAG FEATURES: the dataset is sparse (~36 records on average per
restaurant+item pair, spread irregularly across a year) rather than a
dense daily series. A calendar "yesterday's sales" lag is not meaningful
here because most restaurant+item pairs simply don't have a record for
the literal previous day. Instead, lag/rolling features are built as
*sequence-based* (previous 1-2 recorded transactions for that
restaurant+item, whatever date they fall on), which is a standard and
honest approach for sparse retail data. This is called out explicitly
here and in the metrics page rather than glossed over.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

CATEGORICAL_COLS = ["restaurant_type", "meal_type", "weather_condition"]
ITEM_COL = "menu_item_name"

BASE_NUMERIC_COLS = [
    "typical_ingredient_cost",
    "observed_market_price",
    "actual_selling_price",
    "has_promotion",
    "special_event",
    "year",
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]

LAG_COLS = ["lag_1", "lag_2", "rolling_mean_3"]


class DemandForecaster:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.item_dummies_cols = None
        self.feature_cols = None
        self.scaler = None
        self.models = {}
        self.metrics = {}
        self.feature_importance = {}
        # per (restaurant_id, menu_item_name) most recent known values,
        # used to build the input row for "predict tomorrow" style calls
        self.latest_state = {}

    # ------------------------------------------------------------------
    # Data loading & feature engineering
    # ------------------------------------------------------------------

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

        df["has_promotion"] = df["has_promotion"].astype(int)
        df["special_event"] = df["special_event"].astype(int)

        # Duplicate (date, restaurant, item, meal) rows are collapsed by
        # summing quantity and averaging price fields, so each row below
        # represents one real observation.
        agg = {
            "quantity_sold": "sum",
            "typical_ingredient_cost": "mean",
            "observed_market_price": "mean",
            "actual_selling_price": "mean",
            "has_promotion": "max",
            "special_event": "max",
            "restaurant_type": "first",
            "weather_condition": "first",
        }
        df = (
            df.groupby(["date", "restaurant_id", ITEM_COL, "meal_type"], as_index=False)
            .agg(agg)
        )

        df = df.sort_values(["restaurant_id", ITEM_COL, "date"]).reset_index(drop=True)
        self.df = df
        return df

    def _add_calendar_features(self, df):
        df = df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month_sin"] = np.sin(df["month"] * (2 * np.pi / 12))
        df["month_cos"] = np.cos(df["month"] * (2 * np.pi / 12))
        df["dow_sin"] = np.sin(df["day_of_week"] * (2 * np.pi / 7))
        df["dow_cos"] = np.cos(df["day_of_week"] * (2 * np.pi / 7))
        return df

    def _add_lag_features(self, df):
        df = df.copy()
        grp_cols = ["restaurant_id", ITEM_COL]
        df["lag_1"] = df.groupby(grp_cols)["quantity_sold"].shift(1)
        df["lag_2"] = df.groupby(grp_cols)["quantity_sold"].shift(2)
        df["rolling_mean_3"] = df.groupby(grp_cols)["quantity_sold"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        # For the first 1-2 observations of a restaurant+item, lags are
        # unknown. Fill with that item's overall median rather than 0,
        # since 0 would look like "predicted to sell nothing" noise.
        for col in LAG_COLS:
            df[col] = df.groupby(ITEM_COL)[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df["quantity_sold"].median())
        return df

    def build_features(self, df=None, fit_dummies=False):
        if df is None:
            df = self.df
        df = self._add_calendar_features(df)
        df = self._add_lag_features(df)

        cat_dummies = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
        item_dummies = pd.get_dummies(df[[ITEM_COL]], prefix=ITEM_COL)

        if fit_dummies:
            self.cat_dummies_cols = cat_dummies.columns.tolist()
            self.item_dummies_cols = item_dummies.columns.tolist()
        else:
            cat_dummies = cat_dummies.reindex(columns=self.cat_dummies_cols, fill_value=0)
            item_dummies = item_dummies.reindex(columns=self.item_dummies_cols, fill_value=0)

        features = pd.concat(
            [df[BASE_NUMERIC_COLS + LAG_COLS + ["restaurant_id"]], cat_dummies, item_dummies],
            axis=1,
        )

        if fit_dummies:
            self.feature_cols = features.columns.tolist()
        else:
            features = features.reindex(columns=self.feature_cols, fill_value=0)

        return features, df["quantity_sold"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_models(self):
        df = self.df.sort_values("date").reset_index(drop=True)

        X, y = self.build_features(df, fit_dummies=True)

        # Chronological split: last 20% of dates held out as test set.
        split_date = df["date"].quantile(0.8, interpolation="nearest")
        train_mask = df["date"] <= split_date
        test_mask = ~train_mask

        X_train, X_test = X[train_mask.values], X[test_mask.values]
        y_train, y_test = y[train_mask.values], y[test_mask.values]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        candidates = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            ),
            "lightgbm": LGBMRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1
            ),
        }

        for name, model in candidates.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            preds = np.clip(preds, 0, None)  # sales can't be negative

            self.metrics[name] = {
                "r2": float(r2_score(y_test, preds)),
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
            }
            self.models[name] = model

        # Feature importance from the tree model with the best R^2 among
        # xgboost/random_forest/lightgbm (linear regression coefficients
        # aren't directly comparable in scale, so it's excluded here).
        tree_models = {k: v for k, v in self.metrics.items() if k != "linear"}
        best_tree_name = max(tree_models, key=lambda k: tree_models[k]["r2"])
        self.best_model_name = best_tree_name

        best_model = self.models[best_tree_name]
        importances = getattr(best_model, "feature_importances_", None)
        if importances is not None:
            imp_series = pd.Series(importances, index=self.feature_cols)
            imp_series = imp_series.sort_values(ascending=False).head(15)
            self.feature_importance = {
                "model": best_tree_name,
                "features": imp_series.index.tolist(),
                "importances": [float(v) for v in imp_series.values],
            }

        # Cache latest known state per (restaurant_id, item) for recommendations
        self._build_latest_state(df)

        self._print_summary()

    def _build_latest_state(self, df):
        latest = df.sort_values("date").groupby(["restaurant_id", ITEM_COL]).tail(1)
        self.latest_state = {}
        for _, row in latest.iterrows():
            key = (int(row["restaurant_id"]), row[ITEM_COL])
            self.latest_state[key] = {
                "restaurant_type": row["restaurant_type"],
                "meal_type": row["meal_type"],
                "typical_ingredient_cost": row["typical_ingredient_cost"],
                "observed_market_price": row["observed_market_price"],
                "actual_selling_price": row["actual_selling_price"],
                "lag_1": row["quantity_sold"],
                "lag_2": row.get("lag_1", row["quantity_sold"]),
                "rolling_mean_3": row.get("rolling_mean_3", row["quantity_sold"]),
            }

    def _print_summary(self):
        print("Training complete (chronological 80/20 split)\n" + "-" * 40)
        for name, m in self.metrics.items():
            print(f"{name:15s} R2={m['r2']:.4f}  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}")
        print(f"\nBest tree model (used for recommendations): {self.best_model_name}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.models, os.path.join(MODEL_DIR, "models.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(
            {
                "feature_cols": self.feature_cols,
                "cat_dummies_cols": self.cat_dummies_cols,
                "item_dummies_cols": self.item_dummies_cols,
                "metrics": self.metrics,
                "feature_importance": self.feature_importance,
                "best_model_name": self.best_model_name,
                "latest_state": self.latest_state,
                "restaurants": sorted(self.df["restaurant_id"].unique().tolist()),
                "items": sorted(self.df[ITEM_COL].unique().tolist()),
                "weather_options": sorted(self.df["weather_condition"].unique().tolist()),
                "meal_types": sorted(self.df["meal_type"].unique().tolist()),
                "restaurant_item_map": self._restaurant_item_map(),
            },
            os.path.join(MODEL_DIR, "meta.pkl"),
        )
        print("Models and metadata saved to saved_models/")

    def _restaurant_item_map(self):
        m = {}
        for rid, item in self.df[["restaurant_id", ITEM_COL]].drop_duplicates().values:
            m.setdefault(int(rid), []).append(item)
        return m

    def load(self):
        self.models = joblib.load(os.path.join(MODEL_DIR, "models.pkl"))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        meta = joblib.load(os.path.join(MODEL_DIR, "meta.pkl"))
        self.feature_cols = meta["feature_cols"]
        self.cat_dummies_cols = meta["cat_dummies_cols"]
        self.item_dummies_cols = meta["item_dummies_cols"]
        self.metrics = meta["metrics"]
        self.feature_importance = meta["feature_importance"]
        self.best_model_name = meta["best_model_name"]
        self.latest_state = meta["latest_state"]
        self.restaurants = meta["restaurants"]
        self.items = meta["items"]
        self.weather_options = meta["weather_options"]
        self.meal_types = meta["meal_types"]
        self.restaurant_item_map = meta["restaurant_item_map"]
        return meta

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _row_from_inputs(self, date, restaurant_id, item, meal_type, weather_condition,
                          has_promotion, special_event):
        key = (int(restaurant_id), item)
        state = self.latest_state.get(key)
        if state is None:
            # Unseen restaurant+item combo: fall back to item-level median state
            fallback = [v for k, v in self.latest_state.items() if k[1] == item]
            if not fallback:
                raise ValueError(f"Unknown menu item '{item}'")
            state = fallback[0]

        row = pd.DataFrame([{
            "date": pd.to_datetime(date),
            "restaurant_id": int(restaurant_id),
            ITEM_COL: item,
            "meal_type": meal_type,
            "weather_condition": weather_condition,
            "restaurant_type": state["restaurant_type"],
            "typical_ingredient_cost": state["typical_ingredient_cost"],
            "observed_market_price": state["observed_market_price"],
            "actual_selling_price": state["actual_selling_price"],
            "has_promotion": int(has_promotion),
            "special_event": int(special_event),
            "quantity_sold": np.nan,
        }])

        row = self._add_calendar_features(row)
        for col in LAG_COLS:
            row[col] = state[col]

        cat_dummies = pd.get_dummies(row[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
        cat_dummies = cat_dummies.reindex(columns=self.cat_dummies_cols, fill_value=0)
        item_dummies = pd.get_dummies(row[[ITEM_COL]], prefix=ITEM_COL)
        item_dummies = item_dummies.reindex(columns=self.item_dummies_cols, fill_value=0)

        features = pd.concat(
            [row[BASE_NUMERIC_COLS + LAG_COLS + ["restaurant_id"]], cat_dummies, item_dummies],
            axis=1,
        )
        features = features.reindex(columns=self.feature_cols, fill_value=0)
        return features

    def predict(self, date, restaurant_id, item, meal_type, weather_condition,
                has_promotion=False, special_event=False):
        features = self._row_from_inputs(
            date, restaurant_id, item, meal_type, weather_condition,
            has_promotion, special_event,
        )
        features_scaled = self.scaler.transform(features)

        predictions = {}
        for name, model in self.models.items():
            pred = float(model.predict(features_scaled)[0])
            predictions[name] = max(0.0, round(pred, 1))
        return predictions

    def recommend_for_restaurant(self, date, restaurant_id, weather_condition,
                                  has_promotion=False, special_event=False):
        """Predict tomorrow's demand for every menu item this restaurant sells,
        using the best-performing tree model, sorted by predicted quantity."""
        items = self.restaurant_item_map.get(int(restaurant_id), [])
        results = []
        for item in items:
            key = (int(restaurant_id), item)
            meal_type = self.latest_state[key]["meal_type"]
            preds = self.predict(
                date, restaurant_id, item, meal_type, weather_condition,
                has_promotion, special_event,
            )
            results.append({
                "item": item,
                "meal_type": meal_type,
                "predicted_quantity": preds[self.best_model_name],
            })
        results.sort(key=lambda r: r["predicted_quantity"], reverse=True)
        return results


if __name__ == "__main__":
    forecaster = DemandForecaster("dataset/restaurant_sales_data.csv")
    forecaster.load_data()
    forecaster.train_models()
    forecaster.save()
