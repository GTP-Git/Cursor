"""
Non-linear cruise price forecasting with engineered calendar features and
ensemble tree models (Random Forest, XGBoost, LightGBM).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

CABIN_FORECAST_COLUMNS = (
    "interior_price",
    "oceanview_price",
    "balcony_price",
    "suite_price",
)

MIN_HISTORY_POINTS = 6
FORECAST_HORIZONS_DAYS = (7, 14)
CV_SPLITS = 3

US_HOLIDAYS = holidays.US()

PEAK_SAILING_MONTHS = {3, 6, 7, 8, 12}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Any


def _history_to_frame(history: list[Any]) -> pd.DataFrame:
    if not history:
        return pd.DataFrame()
    df = pd.DataFrame([dict(row) for row in history])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def _parse_sailing_date(cruise: Any) -> pd.Timestamp | None:
    raw = None
    if cruise is not None:
        if isinstance(cruise, dict):
            raw = cruise.get("sailing_date")
        else:
            try:
                raw = cruise["sailing_date"]
            except (KeyError, TypeError):
                raw = getattr(cruise, "sailing_date", None)
    if not raw or str(raw).strip().upper() == "TBD":
        return None
    parsed = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.tz_localize(None) if parsed.tzinfo else parsed


def _parse_nights(cruise: Any) -> int:
    duration = None
    url = None
    if cruise is not None:
        if isinstance(cruise, dict):
            duration = cruise.get("duration")
            url = cruise.get("url")
        else:
            try:
                duration = cruise["duration"]
                url = cruise["url"]
            except (KeyError, TypeError):
                duration = getattr(cruise, "duration", None)
                url = getattr(cruise, "url", None)
    if duration:
        import re

        match = re.search(r"(\d+)", str(duration))
        if match:
            return max(1, int(match.group(1)))
    if url:
        import re

        match = re.search(r"(\d+)-night", str(url).lower())
        if match:
            return max(1, int(match.group(1)))
    return 7


def _is_holiday(ts: pd.Timestamp) -> int:
    day = ts.tz_localize(None).date() if ts.tzinfo else ts.date()
    return int(day in US_HOLIDAYS)


def _calendar_features(ts: pd.Timestamp, sailing_date: pd.Timestamp | None, nights: int) -> dict[str, float]:
    ts = ts.tz_localize(None) if ts.tzinfo else ts
    days_until = (sailing_date - ts).days if sailing_date is not None else np.nan
    if sailing_date is not None and days_until < 0:
        days_until = 0.0

    dow = int(ts.dayofweek)
    month = int(ts.month)
    sailing_month = int(sailing_date.month) if sailing_date is not None else month

    return {
        "days_until_sailing": float(days_until) if not np.isnan(days_until) else 180.0,
        "log_days_until_sailing": float(np.log1p(max(days_until, 0)))
        if not np.isnan(days_until)
        else float(np.log1p(180)),
        "obs_dow": float(dow),
        "obs_is_weekend": float(dow >= 5),
        "obs_is_holiday": float(_is_holiday(ts)),
        "obs_month": float(month),
        "obs_week_of_year": float(ts.isocalendar().week),
        "obs_quarter": float((month - 1) // 3 + 1),
        "sailing_month": float(sailing_month),
        "sailing_is_summer": float(sailing_month in {6, 7, 8}),
        "sailing_is_peak": float(sailing_month in PEAK_SAILING_MONTHS),
        "cruise_nights": float(nights),
        "days_until_within_30": float(1.0 if not np.isnan(days_until) and days_until <= 30 else 0.0),
        "days_until_within_90": float(1.0 if not np.isnan(days_until) and days_until <= 90 else 0.0),
        "days_until_within_180": float(1.0 if not np.isnan(days_until) and days_until <= 180 else 0.0),
    }


def _build_training_frame(
    history: list[Any], cruise: Any, target_column: str
) -> tuple[pd.DataFrame, pd.Series] | None:
    df = _history_to_frame(history)
    if target_column not in df.columns:
        return None

    series = df.dropna(subset=[target_column])
    if len(series) < MIN_HISTORY_POINTS:
        return None

    sailing_date = _parse_sailing_date(cruise)
    nights = _parse_nights(cruise)
    origin = series["timestamp"].iloc[0]
    prices = series[target_column].astype(float)

    rows: list[dict[str, float]] = []
    for idx in range(len(series)):
        ts = series["timestamp"].iloc[idx]
        row = _calendar_features(ts, sailing_date, nights)
        row["days_since_tracking_start"] = float(
            (ts - origin).total_seconds() / 86_400.0
        )
        row["snapshot_index"] = float(idx)
        row["lag_1"] = float(prices.iloc[idx - 1]) if idx >= 1 else float(prices.iloc[idx])
        row["lag_2"] = float(prices.iloc[idx - 2]) if idx >= 2 else float(prices.iloc[idx])
        if idx >= 2:
            row["rolling_mean_3"] = float(prices.iloc[max(0, idx - 2) : idx + 1].mean())
        else:
            row["rolling_mean_3"] = float(prices.iloc[idx])
        first = float(prices.iloc[0])
        current = float(prices.iloc[idx])
        row["pct_from_first"] = float((current - first) / first * 100) if first else 0.0
        row["pct_from_lag1"] = (
            float((current - prices.iloc[idx - 1]) / prices.iloc[idx - 1] * 100)
            if idx >= 1 and prices.iloc[idx - 1]
            else 0.0
        )

        for tier in CABIN_FORECAST_COLUMNS:
            if tier == target_column:
                continue
            val = series[tier].iloc[idx] if tier in series.columns else np.nan
            row[tier] = float(val) if pd.notna(val) else current

        if len(CABIN_FORECAST_COLUMNS) >= 2:
            tier_vals = [
                float(series[c].iloc[idx])
                for c in CABIN_FORECAST_COLUMNS
                if c in series.columns and pd.notna(series[c].iloc[idx])
            ]
            if tier_vals:
                row["tier_min"] = float(min(tier_vals))
                row["tier_max"] = float(max(tier_vals))
                row["tier_spread"] = float(max(tier_vals) - min(tier_vals))
                row["tier_mean"] = float(np.mean(tier_vals))

        rows.append(row)

    feature_df = pd.DataFrame(rows)
    y = prices.reset_index(drop=True)
    return feature_df, y


def _model_candidates() -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            "random_forest",
            lambda: RandomForestRegressor(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
            ),
        ),
    ]

    try:
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                "xgboost",
                lambda: XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                ),
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                "lightgbm",
                lambda: LGBMRegressor(
                    n_estimators=200,
                    max_depth=4,
                    num_leaves=16,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    reg_alpha=0.5,
                    reg_lambda=1.0,
                    min_child_samples=2,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )
    except Exception:
        pass

    return specs


def _evaluate_models(
    X: pd.DataFrame, y: pd.Series
) -> tuple[Any, str, float, dict[str, float]]:
    """Pick the lowest time-series CV MAE among tree models."""
    n_splits = min(CV_SPLITS, max(2, len(X) - 2))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_model = None
    best_name = "random_forest"
    best_mae = float("inf")
    model_maes: dict[str, float] = {}

    for spec in _model_candidates():
        fold_maes: list[float] = []
        for train_idx, test_idx in tscv.split(X):
            if len(test_idx) == 0:
                continue
            model = spec.builder()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            fold_maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        if not fold_maes:
            continue
        avg_mae = float(np.mean(fold_maes))
        model_maes[spec.name] = round(avg_mae, 2)
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_name = spec.name
            best_model = spec.builder()
            best_model.fit(X, y)

    if best_model is None:
        spec = _model_candidates()[0]
        best_model = spec.builder()
        best_model.fit(X, y)
        best_name = spec.name
        best_mae = float(mean_absolute_error(y, best_model.predict(X)))
        model_maes[best_name] = round(best_mae, 2)

    return best_model, best_name, best_mae, model_maes


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return {}
    pairs = sorted(
        zip(feature_names, values), key=lambda item: item[1], reverse=True
    )
    return {name: round(float(score), 4) for name, score in pairs[:8]}


def _future_feature_row(
    history: list[Any],
    cruise: Any,
    target_column: str,
    horizon_days: int,
) -> pd.DataFrame | None:
    built = _build_training_frame(history, cruise, target_column)
    if built is None:
        return None
    train_X, train_y = built
    if train_X.empty:
        return None

    df = _history_to_frame(history)
    series = df.dropna(subset=[target_column]).sort_values("timestamp")
    last_ts = series["timestamp"].iloc[-1]
    future_ts = last_ts + timedelta(days=horizon_days)

    sailing_date = _parse_sailing_date(cruise)
    nights = _parse_nights(cruise)
    origin = series["timestamp"].iloc[0]
    prices = series[target_column].astype(float)
    current = float(prices.iloc[-1])

    row = _calendar_features(future_ts, sailing_date, nights)
    row["days_since_tracking_start"] = float(
        (future_ts - origin).total_seconds() / 86_400.0
    )
    row["snapshot_index"] = float(len(series))
    row["lag_1"] = current
    row["lag_2"] = float(prices.iloc[-2]) if len(prices) >= 2 else current
    row["rolling_mean_3"] = float(prices.tail(3).mean())
    first = float(prices.iloc[0])
    row["pct_from_first"] = float((current - first) / first * 100) if first else 0.0
    row["pct_from_lag1"] = 0.0

    last_row = series.iloc[-1]
    for tier in CABIN_FORECAST_COLUMNS:
        if tier == target_column:
            continue
        val = last_row[tier] if tier in series.columns else np.nan
        row[tier] = float(val) if pd.notna(val) else current

    tier_vals = [
        float(last_row[c])
        for c in CABIN_FORECAST_COLUMNS
        if c in series.columns and pd.notna(last_row[c])
    ]
    if tier_vals:
        row["tier_min"] = float(min(tier_vals))
        row["tier_max"] = float(max(tier_vals))
        row["tier_spread"] = float(max(tier_vals) - min(tier_vals))
        row["tier_mean"] = float(np.mean(tier_vals))

    future_df = pd.DataFrame([row])
    for col in train_X.columns:
        if col not in future_df.columns:
            future_df[col] = 0.0
    return future_df[train_X.columns]


def forecast_cabin_price(
    history: list[Any], cruise: Any, column: str
) -> dict[str, Any] | None:
    """Train tree ensemble and project 7/14-day prices using calendar + lag features."""
    built = _build_training_frame(history, cruise, column)
    if built is None:
        return None
    X, y = built
    if y.nunique() < 2:
        return None

    model, model_name, cv_mae, model_maes = _evaluate_models(X, y)

    df = _history_to_frame(history)
    series = df.dropna(subset=[column]).sort_values("timestamp")
    last_ts = series["timestamp"].iloc[-1]
    current = round(float(y.iloc[-1]), 2)

    forecasts: dict[str, float] = {}
    forecast_dates: dict[str, str] = {}
    for horizon in FORECAST_HORIZONS_DAYS:
        future_X = _future_feature_row(history, cruise, column, horizon)
        if future_X is None:
            continue
        predicted = float(model.predict(future_X)[0])
        forecasts[f"day_{horizon}"] = max(0.0, round(predicted, 2))
        forecast_dates[f"date_{horizon}"] = (last_ts + timedelta(days=horizon)).strftime(
            "%Y-%m-%d"
        )

    if not forecasts:
        return None

    train_preds = model.predict(X)
    train_mae = float(mean_absolute_error(y, train_preds))
    ss_res = float(np.sum((y - train_preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0

    day_7 = forecasts.get("day_7", current)
    trend = "falling" if day_7 < current - 1 else "rising" if day_7 > current + 1 else "flat"

    return {
        "column": column,
        "current": current,
        "model": model_name,
        "cv_mae": round(cv_mae, 2),
        "train_mae": round(train_mae, 2),
        "model_maes": model_maes,
        "top_features": _feature_importance(model, list(X.columns)),
        "trend": trend,
        "r2": round(r2, 3),
        "confidence": _confidence_label(cv_mae, current, len(y), r2),
        "history_points": len(y),
        **forecasts,
        **forecast_dates,
    }


def _confidence_label(cv_mae: float, current: float, n_points: int, r2: float) -> str:
    if n_points < MIN_HISTORY_POINTS:
        return "insufficient data"
    rel_error = cv_mae / current if current > 0 else cv_mae
    if rel_error <= 0.05 and n_points >= 10 and r2 >= 0.5:
        return "high"
    if rel_error <= 0.12 or n_points >= 8:
        return "medium"
    return "low"


def forecast_all_cabins(
    history: list[Any], cruise: Any
) -> dict[str, dict[str, Any]]:
    """Return forecasts for each cabin tier that has enough history."""
    results: dict[str, dict[str, Any]] = {}
    for column in CABIN_FORECAST_COLUMNS:
        forecast = forecast_cabin_price(history, cruise, column)
        if forecast:
            results[column] = forecast
    return results


def build_forecast_chart_frame(
    history: list[Any], cruise: Any, column: str
) -> pd.DataFrame | None:
    """Historical series plus 7/14-day ML projection for charting."""
    df = _history_to_frame(history)
    series = df[["timestamp", column]].dropna(subset=[column])
    if series.empty:
        return None

    forecast = forecast_cabin_price(history, cruise, column)
    if not forecast:
        chart = series.set_index("timestamp")[[column]].copy()
        chart.columns = ["Actual"]
        return chart

    built = _build_training_frame(history, cruise, column)
    if built is None:
        return None
    X, y = built
    model, _, _, _ = _evaluate_models(X, y)

    last_ts = series["timestamp"].iloc[-1]
    actual = series.set_index("timestamp")[[column]].copy()
    actual.columns = ["Actual"]

    projected_rows: list[tuple[pd.Timestamp, float]] = []
    for horizon in FORECAST_HORIZONS_DAYS:
        future_X = _future_feature_row(history, cruise, column, horizon)
        if future_X is None:
            continue
        price = float(model.predict(future_X)[0])
        projected_rows.append((last_ts + timedelta(days=horizon), max(0.0, price)))

    bridge = pd.DataFrame(
        {"Forecast": [float(y.iloc[-1])]},
        index=pd.DatetimeIndex([last_ts], name="timestamp"),
    )
    projected = pd.DataFrame(
        {"Forecast": [p for _, p in projected_rows]},
        index=pd.DatetimeIndex([ts for ts, _ in projected_rows], name="timestamp"),
    )
    return pd.concat([actual, bridge, projected]).sort_index()
