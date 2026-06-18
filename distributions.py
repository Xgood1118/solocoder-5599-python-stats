import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats
from models import Distribution, DataSet


DISTRIBUTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "norm": {
        "display_name": "Normal (Gaussian)",
        "description": "对称钟形分布，由均值和标准差参数化",
        "support": "(-∞, +∞)",
        "param_names": ["loc (mean)", "scale (std)"],
        "default_params": {"loc": 0.0, "scale": 1.0},
    },
    "lognorm": {
        "display_name": "Log-Normal",
        "description": "对数服从正态分布的正随机变量",
        "support": "(0, +∞)",
        "param_names": ["s (shape)", "loc", "scale"],
        "default_params": {"s": 1.0, "loc": 0.0, "scale": 1.0},
    },
    "expon": {
        "display_name": "Exponential",
        "description": "泊松过程中事件发生间隔时间的分布",
        "support": "[0, +∞)",
        "param_names": ["loc", "scale (1/λ)"],
        "default_params": {"loc": 0.0, "scale": 1.0},
    },
    "gamma": {
        "display_name": "Gamma",
        "description": "指数分布和χ²分布的推广",
        "support": "(0, +∞) when loc=0",
        "param_names": ["a (shape)", "loc", "scale"],
        "default_params": {"a": 2.0, "loc": 0.0, "scale": 1.0},
    },
    "beta": {
        "display_name": "Beta",
        "description": "定义在[0,1]上的两参数分布族",
        "support": "[0, 1] when loc=0, scale=1",
        "param_names": ["a", "b", "loc", "scale"],
        "default_params": {"a": 2.0, "b": 5.0, "loc": 0.0, "scale": 1.0},
    },
    "uniform": {
        "display_name": "Uniform (Continuous)",
        "description": "区间内概率密度处处相等的分布",
        "support": "[loc, loc+scale]",
        "param_names": ["loc", "scale"],
        "default_params": {"loc": 0.0, "scale": 1.0},
    },
    "weibull_min": {
        "display_name": "Weibull (Minimum)",
        "description": "可靠性分析中常用的极值分布",
        "support": "[0, +∞) when loc=0",
        "param_names": ["c (shape)", "loc", "scale"],
        "default_params": {"c": 1.5, "loc": 0.0, "scale": 1.0},
    },
    "pareto": {
        "display_name": "Pareto",
        "description": "重尾分布，描述80/20法则现象",
        "support": "[loc+scale, +∞)",
        "param_names": ["b (shape)", "loc", "scale"],
        "default_params": {"b": 3.0, "loc": 0.0, "scale": 1.0},
    },
    "cauchy": {
        "display_name": "Cauchy",
        "description": "无均值无方差的厚尾分布",
        "support": "(-∞, +∞)",
        "param_names": ["loc", "scale"],
        "default_params": {"loc": 0.0, "scale": 1.0},
    },
    "t": {
        "display_name": "Student's t",
        "description": "小样本统计推断的核心分布",
        "support": "(-∞, +∞)",
        "param_names": ["df (degrees of freedom)", "loc", "scale"],
        "default_params": {"df": 10.0, "loc": 0.0, "scale": 1.0},
    },
    "chi2": {
        "display_name": "Chi-Squared (χ²)",
        "description": "k个独立标准正态变量平方和的分布",
        "support": "[0, +∞) when loc=0",
        "param_names": ["df", "loc", "scale"],
        "default_params": {"df": 5.0, "loc": 0.0, "scale": 1.0},
    },
    "f": {
        "display_name": "F (Fisher-Snedecor)",
        "description": "ANOVA和回归显著性检验用分布",
        "support": "[0, +∞) when loc=0",
        "param_names": ["dfn", "dfd", "loc", "scale"],
        "default_params": {"dfn": 5.0, "dfd": 10.0, "loc": 0.0, "scale": 1.0},
    },
}


def _clean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    return arr[mask]


def _get_distribution(dist_name: str) -> Any:
    if not hasattr(stats, dist_name):
        raise ValueError(f"Unknown distribution: {dist_name}")
    return getattr(stats, dist_name)


def _params_to_dict(dist_name: str, params: Tuple) -> Dict[str, float]:
    catalog = DISTRIBUTION_CATALOG.get(dist_name, {})
    pnames = catalog.get("param_names")
    if pnames is None:
        pnames = [f"param_{i}" for i in range(len(params))]
    short_names = [p.split(" ")[0] for p in pnames]
    return {short_names[i]: float(params[i]) for i in range(len(params))}


def _anderson_darling(data: np.ndarray, dist_name: str, params: Tuple) -> Optional[Dict[str, float]]:
    try:
        if dist_name == "norm":
            ad_result = stats.anderson(data, dist="norm")
        elif dist_name == "expon":
            ad_result = stats.anderson(data, dist="expon")
        elif dist_name == "logistic":
            ad_result = stats.anderson(data, dist="logistic")
        elif dist_name == "gumbel":
            ad_result = stats.anderson(data, dist="gumbel")
        elif dist_name in ["lognorm", "gamma", "weibull_min"]:
            ad_result = stats.anderson(data, dist=dist_name if dist_name != "weibull_min" else "weibull")
        else:
            n = len(data)
            sorted_data = np.sort(data)
            dist_obj = _get_distribution(dist_name)
            cdf_vals = dist_obj.cdf(sorted_data, *params)
            cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
            i_arr = np.arange(1, n + 1)
            ad_stat = -n - np.sum((2 * i_arr - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))) / n
            return {"statistic": float(ad_stat), "critical_values": None, "significance_levels": None}
        return {
            "statistic": float(ad_result.statistic),
            "critical_values": [float(x) for x in ad_result.critical_values],
            "significance_levels": [float(x) for x in ad_result.significance_level],
        }
    except Exception as e:
        return None


def fit_distribution(data: Union[List[float], np.ndarray, DataSet],
                     dist_name: str,
                     column_idx: Optional[int] = None) -> Distribution:
    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        arr = data.get_clean_column(col_idx)
    else:
        arr = _clean_array(np.asarray(data, dtype=float))

    if len(arr) < 5:
        raise ValueError("Need at least 5 data points for distribution fitting")

    dist_obj = _get_distribution(dist_name)
    catalog = DISTRIBUTION_CATALOG.get(dist_name, {})

    try:
        params = dist_obj.fit(arr)
    except Exception as e:
        raise RuntimeError(f"MLE fitting failed for {dist_name}: {e}")

    ks_stat, ks_p = stats.kstest(arr, dist_name, args=params)
    ad_result = _anderson_darling(arr, dist_name, params)

    gof = {
        "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
        "ad_test": ad_result,
        "n_points": len(arr),
    }

    param_dict = _params_to_dict(dist_name, params)
    dist = Distribution(
        name=dist_name,
        parameters=param_dict,
        fitted=True,
        support=catalog.get("support"),
        description=catalog.get("description"),
        goodness_of_fit=gof,
    )
    return dist


def auto_fit_distributions(data: Union[List[float], np.ndarray, DataSet],
                           candidates: Optional[List[str]] = None,
                           top_k: int = 3,
                           column_idx: Optional[int] = None,
                           rank_by: str = "ad_statistic") -> Dict[str, Any]:
    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        arr = data.get_clean_column(col_idx)
        nan_info = {"total_count": data.n_rows, "valid_count": len(arr), "skipped_nan": data.n_rows - len(arr)}
    else:
        arr_raw = np.asarray(data, dtype=float)
        arr = _clean_array(arr_raw)
        nan_info = {"total_count": len(arr_raw), "valid_count": len(arr), "skipped_nan": len(arr_raw) - len(arr)}

    if candidates is None:
        candidates = ["norm", "lognorm", "expon", "gamma", "weibull_min", "beta", "uniform", "t", "cauchy"]

    fitted: List[Distribution] = []
    errors: List[Dict[str, str]] = []
    for dname in candidates:
        try:
            dist = fit_distribution(arr, dname)
            fitted.append(dist)
        except Exception as e:
            errors.append({"distribution": dname, "error": str(e)})

    def rank_key(d: Distribution):
        gof = d.goodness_of_fit or {}
        if rank_by == "ks_p_value":
            ks = gof.get("ks_test", {})
            return -(ks.get("p_value", 0) or 0)
        elif rank_by == "ks_statistic":
            ks = gof.get("ks_test", {})
            return ks.get("statistic", float("inf")) or float("inf")
        else:
            ad = gof.get("ad_test", {})
            return ad.get("statistic", float("inf")) if ad else float("inf")

    fitted.sort(key=rank_key)
    top = fitted[:top_k]

    ranked_list = []
    for rank, d in enumerate(top, start=1):
        ddict = d.to_dict()
        ddict["rank"] = rank
        ranked_list.append(ddict)

    return {
        "skipna": nan_info,
        "n_valid": len(arr),
        "rank_by": rank_by,
        "candidates_tried": candidates,
        "top_k": top_k,
        "top_distributions": ranked_list,
        "errors": errors,
        "catalog_info": {name: info for name, info in DISTRIBUTION_CATALOG.items() if name in candidates},
    }


def _build_args(param_dict: Dict[str, float], dist_name: str) -> Tuple:
    catalog = DISTRIBUTION_CATALOG.get(dist_name, {})
    pnames = catalog.get("param_names")
    if pnames is None:
        return tuple(param_dict[k] for k in sorted(param_dict.keys()))
    short_names = [p.split(" ")[0] for p in pnames]
    return tuple(param_dict[name] for name in short_names if name in param_dict)


def distribution_pdf(dist_name: str, x: Union[float, List[float]],
                     parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    result = dist_obj.pdf(xs, *args)
    return {
        "distribution": dist_name,
        "parameters": params,
        "x": [float(v) for v in xs],
        "pdf": [float(v) for v in result],
    }


def distribution_cdf(dist_name: str, x: Union[float, List[float]],
                     parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    result = dist_obj.cdf(xs, *args)
    return {
        "distribution": dist_name,
        "parameters": params,
        "x": [float(v) for v in xs],
        "cdf": [float(v) for v in result],
    }


def distribution_ppf(dist_name: str, q: Union[float, List[float]],
                     parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    if np.any((qs < 0) | (qs > 1)):
        raise ValueError("Quantiles q must be in [0, 1]")
    result = dist_obj.ppf(qs, *args)
    return {
        "distribution": dist_name,
        "parameters": params,
        "q": [float(v) for v in qs],
        "ppf": [float(v) for v in result],
    }


def distribution_sf(dist_name: str, x: Union[float, List[float]],
                    parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    result = dist_obj.sf(xs, *args)
    return {
        "distribution": dist_name,
        "parameters": params,
        "x": [float(v) for v in xs],
        "sf (survival = 1 - CDF)": [float(v) for v in result],
    }


def distribution_isf(dist_name: str, q: Union[float, List[float]],
                     parameters: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    qs = np.atleast_1d(np.asarray(q, dtype=float))
    if np.any((qs < 0) | (qs > 1)):
        raise ValueError("Survival quantiles q must be in [0, 1]")
    result = dist_obj.isf(qs, *args)
    return {
        "distribution": dist_name,
        "parameters": params,
        "q": [float(v) for v in qs],
        "isf (inverse survival)": [float(v) for v in result],
    }


def distribution_rvs(dist_name: str, size: int = 100,
                     parameters: Optional[Dict[str, float]] = None,
                     random_state: Optional[int] = None) -> Dict[str, Any]:
    dist_obj = _get_distribution(dist_name)
    params = parameters or DISTRIBUTION_CATALOG.get(dist_name, {}).get("default_params", {})
    args = _build_args(params, dist_name)
    result = dist_obj.rvs(*args, size=size, random_state=random_state)
    return {
        "distribution": dist_name,
        "parameters": params,
        "size": size,
        "random_state": random_state,
        "samples": [float(v) for v in result],
        "sample_mean": float(np.mean(result)),
        "sample_std": float(np.std(result, ddof=1)) if size > 1 else None,
    }


def list_distributions() -> Dict[str, Any]:
    return {
        "count": len(DISTRIBUTION_CATALOG),
        "distributions": [
            {
                "key": name,
                **info,
            }
            for name, info in DISTRIBUTION_CATALOG.items()
        ],
    }
