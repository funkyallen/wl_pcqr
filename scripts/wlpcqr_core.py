from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
except Exception:  # pragma: no cover
    NGBRegressor = None
    Normal = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-8

FEATURES = [
    "DEPTH",
    "DTC",
    "DTS",
    "BS",
    "CALI",
    "DEN",
    "DENC",
    "GR",
    "NEU",
    "PEF",
    "RDEP_LOG10",
    "RMED_LOG10",
    "ROP",
]
ACTIVE_FEATURES = list(FEATURES)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = np.sort(scores[np.isfinite(scores)])
    if scores.size == 0:
        return 0.0
    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(scores[k - 1])


def assign_depth_blocks(n_rows: int, n_blocks: int) -> np.ndarray:
    blocks = np.floor(np.arange(n_rows) * n_blocks / max(n_rows, 1)).astype(int)
    return np.clip(blocks, 0, n_blocks - 1)


def load_dataset(path: Path, target: str, max_per_well: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path).replace(-9999.0, np.nan)
    if target not in df.columns:
        raise ValueError(f"Target column is missing: {target}")
    df = df[df[target].notna()].copy()
    df["RDEP_LOG10"] = np.where(df["RDEP"] > 0, np.log10(df["RDEP"]), np.nan)
    df["RMED_LOG10"] = np.where(df["RMED"] > 0, np.log10(df["RMED"]), np.nan)
    df = df[(df[target] >= 0.0) & (df[target] <= 1.0)].copy()
    df = df.sort_values(["WELLNUM", "DEPTH"]).reset_index(drop=True)
    if max_per_well and max_per_well > 0:
        parts: list[pd.DataFrame] = []
        rng = np.random.default_rng(seed)
        for _, group in df.groupby("WELLNUM", sort=True):
            if len(group) <= max_per_well:
                parts.append(group)
                continue
            idx = np.linspace(0, len(group) - 1, max_per_well).round().astype(int)
            idx = np.unique(idx)
            if len(idx) < max_per_well:
                pool = np.setdiff1d(np.arange(len(group)), idx)
                extra = rng.choice(pool, size=max_per_well - len(idx), replace=False)
                idx = np.sort(np.concatenate([idx, extra]))
            parts.append(group.iloc[idx])
        df = pd.concat(parts, ignore_index=True).sort_values(["WELLNUM", "DEPTH"]).reset_index(drop=True)
    return df


def make_point_model(family: str, seed: int):
    if family == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=160,
                min_samples_leaf=8,
                max_features=0.75,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if family == "et":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=180,
                min_samples_leaf=8,
                max_features=0.75,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if family == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.05,
                max_iter=110,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unsupported point family: {family}")


def make_hgb_quantile_model(quantile: float, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile),
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        ),
    )


def make_lightgbm_quantile_model(quantile: float, seed: int):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    )


def make_lightgbm_point_model(seed: int):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMRegressor(
            objective="regression",
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
    )


def make_catboost_quantile_model(quantile: float, seed: int):
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        CatBoostRegressor(
            loss_function=f"Quantile:alpha={float(quantile)}",
            iterations=220,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
        ),
    )


def make_catboost_point_model(seed: int):
    if CatBoostRegressor is None:
        raise RuntimeError("catboost is not installed")
    return make_pipeline(
        SimpleImputer(strategy="median"),
        CatBoostRegressor(
            loss_function="RMSE",
            iterations=220,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
        ),
    )


def fit_predict_point(train: pd.DataFrame, test: pd.DataFrame, target: str, family: str, seed: int) -> tuple[np.ndarray, object]:
    model = make_point_model(family, seed)
    model.fit(train[ACTIVE_FEATURES], train[target])
    pred = np.asarray(model.predict(test[ACTIVE_FEATURES]), dtype=float)
    return pred, model


def fit_predict_hgb_quantile_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_model = make_hgb_quantile_model(alpha / 2.0, seed)
    mid_model = make_point_model("hgb", seed + 1)
    upper_model = make_hgb_quantile_model(1.0 - alpha / 2.0, seed + 2)
    lower_model.fit(train[ACTIVE_FEATURES], train[target])
    mid_model.fit(train[ACTIVE_FEATURES], train[target])
    upper_model.fit(train[ACTIVE_FEATURES], train[target])
    lower = np.asarray(lower_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    pred = np.asarray(mid_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    upper = np.asarray(upper_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    return np.minimum(lower, upper), pred, np.maximum(lower, upper)


def fit_predict_lightgbm_quantile_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_model = make_lightgbm_quantile_model(alpha / 2.0, seed)
    mid_model = make_lightgbm_point_model(seed + 1)
    upper_model = make_lightgbm_quantile_model(1.0 - alpha / 2.0, seed + 2)
    lower_model.fit(train[ACTIVE_FEATURES], train[target])
    mid_model.fit(train[ACTIVE_FEATURES], train[target])
    upper_model.fit(train[ACTIVE_FEATURES], train[target])
    lower = np.asarray(lower_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    pred = np.asarray(mid_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    upper = np.asarray(upper_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    return np.minimum(lower, upper), pred, np.maximum(lower, upper)


def fit_predict_catboost_quantile_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_model = make_catboost_quantile_model(alpha / 2.0, seed)
    mid_model = make_catboost_point_model(seed + 1)
    upper_model = make_catboost_quantile_model(1.0 - alpha / 2.0, seed + 2)
    lower_model.fit(train[ACTIVE_FEATURES], train[target])
    mid_model.fit(train[ACTIVE_FEATURES], train[target])
    upper_model.fit(train[ACTIVE_FEATURES], train[target])
    lower = np.asarray(lower_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    pred = np.asarray(mid_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    upper = np.asarray(upper_model.predict(test[ACTIVE_FEATURES]), dtype=float)
    return np.minimum(lower, upper), pred, np.maximum(lower, upper)


def fit_predict_tree_quantiles(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    family: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = make_point_model(family, seed)
    model.fit(train[ACTIVE_FEATURES], train[target])
    pred = np.asarray(model.predict(test[ACTIVE_FEATURES]), dtype=float)
    x_test = model.named_steps["simpleimputer"].transform(test[ACTIVE_FEATURES])
    forest = model.named_steps["extratreesregressor"] if family == "et" else model.named_steps["randomforestregressor"]
    tree_preds = np.asarray([tree.predict(x_test) for tree in forest.estimators_], dtype=float)
    lower = np.quantile(tree_preds, alpha / 2.0, axis=0)
    upper = np.quantile(tree_preds, 1.0 - alpha / 2.0, axis=0)
    return pred, np.minimum(lower, upper), np.maximum(lower, upper)


def fit_predict_ngboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if NGBRegressor is None or Normal is None:
        raise RuntimeError("ngboost is not installed")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        NGBRegressor(
            Dist=Normal,
            n_estimators=160,
            learning_rate=0.025,
            natural_gradient=False,
            random_state=seed,
            verbose=False,
        ),
    )
    model.fit(train[ACTIVE_FEATURES], train[target])
    transformed = model.named_steps["standardscaler"].transform(
        model.named_steps["simpleimputer"].transform(test[ACTIVE_FEATURES])
    )
    dist = model.named_steps["ngbregressor"].pred_dist(transformed)
    pred = np.asarray(dist.mean(), dtype=float)
    lower = np.asarray(dist.ppf(alpha / 2.0), dtype=float)
    upper = np.asarray(dist.ppf(1.0 - alpha / 2.0), dtype=float)
    return pred, np.minimum(lower, upper), np.maximum(lower, upper)


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    corr = np.nan
    if y.size > 2 and np.nanstd(y) > 0 and np.nanstd(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "corr": corr,
    }


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    lower = np.minimum(lower_arr, upper_arr)
    upper = np.maximum(lower_arr, upper_arr)
    width = upper - lower
    below = y < lower
    above = y > upper
    winkler = width.copy()
    winkler[below] += (2.0 / alpha) * (lower[below] - y[below])
    winkler[above] += (2.0 / alpha) * (y[above] - upper[above])
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "width": float(np.mean(width)),
        "width_std": float(np.std(width)),
        "width_cv": float(np.std(width) / max(float(np.mean(width)), EPS)),
        "winkler": float(np.mean(winkler)),
    }


def clip_interval(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
