import numpy as np
from typing import List, Dict, Any, Optional, Union
from scipy import stats
from models import DataSet


def _clean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    return arr[mask]


def _skipna_info(arr: np.ndarray) -> Dict[str, int]:
    total = len(arr)
    clean = _clean_array(arr)
    return {"total_count": total, "valid_count": len(clean), "skipped_nan": total - len(clean)}


def descriptive_univariate(
    data: Union[List[float], np.ndarray, DataSet],
    ddof: int = 0,
    quantiles: Optional[List[float]] = None,
    column_idx: Optional[int] = None
) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        arr = data.get_clean_column(col_idx)
        skipna = {"total_count": data.n_rows, "valid_count": len(arr), "skipped_nan": data.n_rows - len(arr)}
    else:
        arr_raw = np.asarray(data, dtype=float)
        arr = _clean_array(arr_raw)
        skipna = _skipna_info(arr_raw)

    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75, 0.9, 0.99]

    result: Dict[str, Any] = {"skipna": skipna}

    if len(arr) == 0:
        return {**result, "mean": None, "median": None, "std_sample": None, "std_population": None,
                "variance_sample": None, "variance_population": None, "min": None, "max": None,
                "range": None, "quantiles": {}, "skew": None, "kurtosis": None, "count": 0}

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_pop = float(np.std(arr, ddof=0))
    std_sample = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
    var_pop = float(np.var(arr, ddof=0))
    var_sample = float(np.var(arr, ddof=1)) if len(arr) > 1 else None
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    quantile_results: Dict[str, float] = {}
    for q in quantiles:
        qname = f"P{int(q * 100)}" if (q * 100).is_integer() else f"P{q * 100:.1f}"
        quantile_results[qname] = float(np.quantile(arr, q))

    skew_val = float(stats.skew(arr, bias=False)) if len(arr) >= 3 else None
    kurt_val = float(stats.kurtosis(arr, fisher=True, bias=False)) if len(arr) >= 4 else None

    return {
        **result,
        "count": len(arr),
        "mean": mean_val,
        "median": median_val,
        "std_population": std_pop,
        "std_sample": std_sample,
        "variance_population": var_pop,
        "variance_sample": var_sample,
        "min": min_val,
        "max": max_val,
        "range": max_val - min_val,
        "quantiles": quantile_results,
        "skew": skew_val,
        "kurtosis": kurt_val,
    }


def _corr_pair(x: np.ndarray, y: np.ndarray, method: str) -> Dict[str, Any]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    xc, yc = x[mask], y[mask]
    n_valid = int(mask.sum())
    info = {"total_count": len(x), "valid_count": n_valid, "skipped_nan": int(len(x) - n_valid)}

    if n_valid < 3:
        return {"correlation": None, "skipna": info, "p_value": None}

    if method == "pearson":
        r, p = stats.pearsonr(xc, yc)
    elif method == "spearman":
        r, p = stats.spearmanr(xc, yc)
    elif method == "kendall":
        r, p = stats.kendalltau(xc, yc)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return {"correlation": float(r), "skipna": info, "p_value": float(p)}


def correlation_matrix(
    data: Union[List[List[float]], np.ndarray, DataSet],
    method: str = "pearson"
) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        columns = [data.get_clean_column(i) for i in range(data.n_cols)]
        column_names = data.column_names
        raw_cols = [data.get_raw_column(i) for i in range(data.n_cols)]
    else:
        mat = np.asarray(data, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        raw_cols = [mat[:, i] for i in range(mat.shape[1])]
        columns = [_clean_array(c) for c in raw_cols]
        column_names = [f"col_{i}" for i in range(len(columns))]

    n = len(columns)
    corr_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)
    skipna_info: Dict[str, Dict] = {}

    for i in range(n):
        for j in range(i, n):
            pair_res = _corr_pair(raw_cols[i], raw_cols[j], method)
            corr_mat[i, j] = pair_res["correlation"] if pair_res["correlation"] is not None else np.nan
            corr_mat[j, i] = corr_mat[i, j]
            p_mat[i, j] = pair_res["p_value"] if pair_res["p_value"] is not None else np.nan
            p_mat[j, i] = p_mat[i, j]
            key = f"{column_names[i]}__{column_names[j]}"
            skipna_info[key] = pair_res["skipna"]

    def _mat_to_list(m):
        return [[None if np.isnan(v) else float(v) for v in row] for row in m]

    return {
        "method": method,
        "column_names": column_names,
        "correlation_matrix": _mat_to_list(corr_mat),
        "p_value_matrix": _mat_to_list(p_mat),
        "pairwise_skipna": skipna_info,
    }


def covariance_matrix(
    data: Union[List[List[float]], np.ndarray, DataSet],
    ddof: int = 1
) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        raw_cols = [data.get_raw_column(i) for i in range(data.n_cols)]
        column_names = data.column_names
    else:
        mat = np.asarray(data, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        raw_cols = [mat[:, i] for i in range(mat.shape[1])]
        column_names = [f"col_{i}" for i in range(len(raw_cols))]

    n = len(raw_cols)
    cov_mat = np.full((n, n), np.nan)
    skipna_info: Dict[str, Dict] = {}

    for i in range(n):
        for j in range(i, n):
            mask = ~np.isnan(raw_cols[i]) & ~np.isnan(raw_cols[j])
            n_valid = int(mask.sum())
            info = {"total_count": len(raw_cols[i]), "valid_count": n_valid, "skipped_nan": int(len(raw_cols[i]) - n_valid)}
            key = f"{column_names[i]}__{column_names[j]}"
            skipna_info[key] = info
            if n_valid > ddof:
                xi = raw_cols[i][mask]
                xj = raw_cols[j][mask]
                cov = float(np.cov(xi, xj, ddof=ddof)[0, 1])
                cov_mat[i, j] = cov
                cov_mat[j, i] = cov

    def _mat_to_list(m):
        return [[None if np.isnan(v) else float(v) for v in row] for row in m]

    return {
        "ddof": ddof,
        "column_names": column_names,
        "covariance_matrix": _mat_to_list(cov_mat),
        "pairwise_skipna": skipna_info,
    }


def summary_stats(
    data: Union[List[List[float]], np.ndarray, DataSet],
    ddof: int = 0,
    quantiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        n_cols = data.n_cols
        col_names = data.column_names
        per_column = {
            col_names[i]: descriptive_univariate(data, ddof=ddof, quantiles=quantiles, column_idx=i)
            for i in range(n_cols)
        }
    else:
        mat = np.asarray(data, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        n_cols = mat.shape[1]
        col_names = [f"col_{i}" for i in range(n_cols)]
        per_column = {
            col_names[i]: descriptive_univariate(mat[:, i], ddof=ddof, quantiles=quantiles)
            for i in range(n_cols)
        }

    return {
        "column_names": col_names,
        "num_columns": n_cols,
        "per_column": per_column,
    }
