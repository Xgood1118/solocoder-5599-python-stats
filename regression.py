import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from models import RegressionModel, DataSet


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
os.makedirs(MODELS_DIR, exist_ok=True)


def _clean_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_flat = y.reshape(-1)
    row_has_nan = np.any(np.isnan(X), axis=1) | np.isnan(y_flat)
    mask = ~row_has_nan
    total = len(X)
    valid = int(mask.sum())
    return X[mask], y_flat[mask], {"total_count": total, "valid_count": valid, "skipped_nan_rows": total - valid}


def _make_regression_result(model, model_type: str, hyperparams: Dict,
                           X: np.ndarray, y: np.ndarray, skipna: Dict,
                           feature_names: Optional[List[str]], target_name: Optional[str]) -> Dict[str, Any]:
    y_pred = model.predict(X)
    params: Dict[str, Any] = {}

    if isinstance(model, Pipeline):
        reg = model.named_steps["regressor"]
        poly = model.named_steps.get("polynomial")
        if poly is not None:
            params["polynomial_degree"] = poly.degree
            params["n_polynomial_features"] = poly.n_output_features_
    else:
        reg = model

    if hasattr(reg, "coef_"):
        params["coefficients"] = reg.coef_.tolist() if hasattr(reg.coef_, "tolist") else list(reg.coef_)
    if hasattr(reg, "intercept_"):
        params["intercept"] = float(reg.intercept_) if np.ndim(reg.intercept_) == 0 else reg.intercept_.tolist()
    if hasattr(reg, "classes_"):
        params["classes"] = reg.classes_.tolist()

    n, p = X.shape[0], X.shape[1]

    if model_type == "logistic":
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        }
        if len(np.unique(y)) == 2 and hasattr(reg, "predict_proba"):
            try:
                metrics["roc_auc"] = float(roc_auc_score(y, reg.predict_proba(X)[:, 1]))
            except Exception:
                pass
    else:
        mse = float(mean_squared_error(y, y_pred))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r_squared": float(r2_score(y, y_pred)),
            "adj_r_squared": float(1 - (1 - r2_score(y, y_pred)) * (n - 1) / (n - p - 1)) if n > p + 1 else None,
        }

    wrap = RegressionModel(
        model_type=model_type,
        parameters=params,
        features=feature_names,
        target=target_name,
        metrics=metrics,
        hyperparameters=hyperparams,
        fitted=True,
        trained_at=datetime.now().isoformat(),
        metadata={"skipna": skipna, "n_samples": n, "n_features": p},
    )
    return {"model": wrap, "sklearn_model": model, "info": wrap.to_dict()}


def fit_linear(X: Union[List, np.ndarray, DataSet],
               y: Union[List, np.ndarray],
               fit_intercept: bool = True,
               feature_names: Optional[List[str]] = None,
               target_name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
        if feature_names is None:
            feature_names = X.column_names
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_clean, y_clean)
    return _make_regression_result(model, "linear", {"fit_intercept": fit_intercept},
                                   X_clean, y_clean, skipna, feature_names, target_name)


def fit_polynomial(X: Union[List, np.ndarray, DataSet],
                   y: Union[List, np.ndarray],
                   degree: int = 2,
                   interaction_only: bool = False,
                   fit_intercept: bool = True,
                   feature_names: Optional[List[str]] = None,
                   target_name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
        if feature_names is None:
            feature_names = X.column_names
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    model = Pipeline([
        ("polynomial", PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)),
        ("regressor", LinearRegression(fit_intercept=fit_intercept)),
    ])
    model.fit(X_clean, y_clean)
    return _make_regression_result(model, "polynomial",
                                   {"degree": degree, "interaction_only": interaction_only, "fit_intercept": fit_intercept},
                                   X_clean, y_clean, skipna, feature_names, target_name)


def fit_ridge(X: Union[List, np.ndarray, DataSet],
              y: Union[List, np.ndarray],
              alpha: float = 1.0,
              fit_intercept: bool = True,
              normalize: bool = False,
              feature_names: Optional[List[str]] = None,
              target_name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
        if feature_names is None:
            feature_names = X.column_names
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    steps = []
    if normalize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", Ridge(alpha=alpha, fit_intercept=fit_intercept)))
    model = Pipeline(steps)
    model.fit(X_clean, y_clean)
    return _make_regression_result(model, "ridge",
                                   {"alpha": alpha, "fit_intercept": fit_intercept, "normalize": normalize},
                                   X_clean, y_clean, skipna, feature_names, target_name)


def fit_lasso(X: Union[List, np.ndarray, DataSet],
              y: Union[List, np.ndarray],
              alpha: float = 1.0,
              fit_intercept: bool = True,
              normalize: bool = False,
              max_iter: int = 10000,
              feature_names: Optional[List[str]] = None,
              target_name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
        if feature_names is None:
            feature_names = X.column_names
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    steps = []
    if normalize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)))
    model = Pipeline(steps)
    model.fit(X_clean, y_clean)
    reg = model.named_steps["regressor"]
    result = _make_regression_result(model, "lasso",
                                     {"alpha": alpha, "fit_intercept": fit_intercept,
                                      "normalize": normalize, "max_iter": max_iter},
                                     X_clean, y_clean, skipna, feature_names, target_name)
    coef_arr = np.asarray(reg.coef_)
    result["info"]["parameters"]["n_nonzero_coefficients"] = int(np.sum(coef_arr != 0))
    if feature_names:
        result["info"]["parameters"]["selected_features"] = [
            name for name, c in zip(feature_names, coef_arr) if c != 0
        ]
    return result


def fit_logistic(X: Union[List, np.ndarray, DataSet],
                 y: Union[List, np.ndarray],
                 C: float = 1.0,
                 penalty: str = "l2",
                 solver: str = "lbfgs",
                 max_iter: int = 1000,
                 fit_intercept: bool = True,
                 feature_names: Optional[List[str]] = None,
                 target_name: Optional[str] = None) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
        if feature_names is None:
            feature_names = X.column_names
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, fit_intercept=fit_intercept)
    model.fit(X_clean, y_clean)
    result = _make_regression_result(model, "logistic",
                                     {"C": C, "penalty": penalty, "solver": solver,
                                      "max_iter": max_iter, "fit_intercept": fit_intercept},
                                     X_clean, y_clean, skipna, feature_names, target_name)
    return result


def predict_model(sklearn_model: Any, X: Union[List, np.ndarray, DataSet]) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    mask = ~np.any(np.isnan(X_arr), axis=1)
    nan_info = {"total_count": len(X_arr), "valid_count": int(mask.sum()), "skipped_nan_rows": len(X_arr) - int(mask.sum())}

    X_valid = X_arr[mask]
    if len(X_valid) == 0:
        return {"predictions": [], "skipna": nan_info, "note": "No valid rows"}

    preds = sklearn_model.predict(X_valid)
    full_preds: List[Optional[float]] = [None] * len(X_arr)
    idx = 0
    for i, m in enumerate(mask):
        if m:
            full_preds[i] = float(preds[idx])
            idx += 1

    result = {"predictions": full_preds, "skipna": nan_info, "valid_predictions": preds.tolist()}

    if hasattr(sklearn_model, "predict_proba"):
        try:
            proba = sklearn_model.predict_proba(X_valid)
            result["probabilities"] = proba.tolist()
        except Exception:
            pass
    return result


def score_model(sklearn_model: Any, X: Union[List, np.ndarray, DataSet],
                y: Union[List, np.ndarray]) -> Dict[str, Any]:
    if isinstance(X, DataSet):
        X_arr = np.asarray(X.data, dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    X_clean, y_clean, skipna = _clean_xy(X_arr, y_arr)
    y_pred = sklearn_model.predict(X_clean)
    model_type = "logistic" if isinstance(sklearn_model, LogisticRegression) or (
        isinstance(sklearn_model, Pipeline) and isinstance(sklearn_model.named_steps.get("regressor"), LogisticRegression)
    ) else "regression"

    if model_type == "logistic":
        metrics = {
            "accuracy": float(accuracy_score(y_clean, y_pred)),
            "precision_macro": float(precision_score(y_clean, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_clean, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_clean, y_pred, average="macro", zero_division=0)),
        }
    else:
        mse = float(mean_squared_error(y_clean, y_pred))
        metrics = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_clean, y_pred)),
            "r_squared": float(r2_score(y_clean, y_pred)),
        }
    return {"model_type": model_type, "metrics": metrics, "skipna": skipna}


def save_model(sklearn_model: Any, model_info: RegressionModel,
               filepath: Optional[str] = None) -> Dict[str, Any]:
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, f"{model_info.model_id}.pkl")
    payload = {
        "sklearn_model": sklearn_model,
        "model_info": model_info,
        "saved_at": datetime.now().isoformat(),
    }
    with open(filepath, "wb") as f:
        pickle.dump(payload, f)
    return {
        "saved": True,
        "filepath": os.path.abspath(filepath),
        "model_id": model_info.model_id,
        "model_type": model_info.model_type,
        "filesize_bytes": os.path.getsize(filepath),
    }


def load_model(filepath: str) -> Dict[str, Any]:
    if not os.path.isabs(filepath):
        filepath = os.path.join(MODELS_DIR, filepath)
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    model_info: RegressionModel = payload["model_info"]
    return {
        "loaded": True,
        "filepath": os.path.abspath(filepath),
        "sklearn_model": payload["sklearn_model"],
        "model_info": model_info,
        "info": model_info.to_dict(),
        "saved_at": payload.get("saved_at"),
    }


def list_saved_models() -> Dict[str, Any]:
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    infos = []
    for fname in files:
        fp = os.path.join(MODELS_DIR, fname)
        try:
            with open(fp, "rb") as f:
                payload = pickle.load(f)
            mi: RegressionModel = payload["model_info"]
            infos.append({
                "filename": fname,
                "filepath": os.path.abspath(fp),
                "model_id": mi.model_id,
                "model_type": mi.model_type,
                "fitted": mi.fitted,
                "trained_at": mi.trained_at,
                "saved_at": payload.get("saved_at"),
                "filesize_bytes": os.path.getsize(fp),
            })
        except Exception:
            infos.append({"filename": fname, "filepath": os.path.abspath(fp), "error": "unreadable"})
    return {"count": len(infos), "models_dir": os.path.abspath(MODELS_DIR), "models": infos}


def delete_model(filepath: str) -> Dict[str, Any]:
    if not os.path.isabs(filepath):
        filepath = os.path.join(MODELS_DIR, filepath)
    if not os.path.exists(filepath):
        return {"deleted": False, "reason": "not_found", "filepath": filepath}
    os.remove(filepath)
    return {"deleted": True, "filepath": os.path.abspath(filepath)}
