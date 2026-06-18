import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from statistics_core import descriptive_univariate
from distributions import auto_fit_distributions
from hypothesis_tests import shapiro_wilk
from models import DataSet


def _clean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    return arr[mask]


def full_statistics_report(data: Union[List[float], np.ndarray, DataSet],
                           column_idx: Optional[int] = None,
                           quantiles: Optional[List[float]] = None,
                           top_k: int = 3,
                           fit_candidates: Optional[List[str]] = None,
                           alpha: float = 0.05) -> Dict[str, Any]:
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        col_name = data.column_names[col_idx]
        unit = data.units.get(col_name)
    else:
        col_name = "value"
        unit = None

    desc = descriptive_univariate(data, ddof=0, quantiles=quantiles, column_idx=column_idx)

    warnings: List[str] = []
    try:
        fit_result = auto_fit_distributions(data, candidates=fit_candidates, top_k=top_k, column_idx=column_idx)
    except Exception as e:
        fit_result = {"error": str(e), "top_distributions": []}
        warnings.append(f"Distribution fitting failed: {e}")

    try:
        sw = shapiro_wilk(data, alpha=alpha, column_idx=column_idx)
    except Exception as e:
        sw = {"error": str(e)}
        warnings.append(f"Shapiro-Wilk test failed: {e}")

    top_dists = fit_result.get("top_distributions", [])
    best_dist = top_dists[0] if top_dists else None

    data_integrity = desc.get("skipna", {})
    n_total = data_integrity.get("total_count", 0)
    n_valid = data_integrity.get("valid_count", 0)
    completion_rate = float(n_valid / n_total * 100) if n_total > 0 else 0.0

    skew_val = desc.get("skew")
    kurt_val = desc.get("kurtosis")
    shape_summary: Dict[str, Any] = {
        "skew_interpretation": None,
        "kurtosis_interpretation": None,
    }
    if skew_val is not None:
        if abs(skew_val) < 0.5:
            shape_summary["skew_interpretation"] = "approximately symmetric"
        elif skew_val > 0:
            shape_summary["skew_interpretation"] = "right-skewed (positive skew)" if skew_val < 1 else "highly right-skewed"
        else:
            shape_summary["skew_interpretation"] = "left-skewed (negative skew)" if skew_val > -1 else "highly left-skewed"
    if kurt_val is not None:
        if abs(kurt_val) < 0.5:
            shape_summary["kurtosis_interpretation"] = "mesokurtic (close to normal)"
        elif kurt_val > 0:
            shape_summary["kurtosis_interpretation"] = "leptokurtic (heavy tails, sharp peak)"
        else:
            shape_summary["kurtosis_interpretation"] = "platykurtic (light tails, flat peak)"

    mean_val = desc.get("mean")
    median_val = desc.get("median")
    std_pop = desc.get("std_population")
    if mean_val and median_val and std_pop:
        cv = float(abs(std_pop / mean_val)) if mean_val != 0 else None
        mean_median_diff_pct = float(abs(mean_val - median_val) / (abs(mean_val) if mean_val != 0 else 1) * 100)
    else:
        cv = None
        mean_median_diff_pct = None

    report_cards = {
        "data_integrity": {
            **data_integrity,
            "completion_rate_percent": completion_rate,
            "status": "OK" if completion_rate >= 90 else ("CAUTION" if completion_rate >= 70 else "POOR"),
        },
        "central_tendency": {
            "mean": mean_val,
            "median": median_val,
            "mode_hint": None,
            "mean_median_diff_pct": mean_median_diff_pct,
            "coefficient_of_variation": cv,
        },
        "dispersion": {
            "min": desc.get("min"),
            "max": desc.get("max"),
            "range": desc.get("range"),
            "std_population": std_pop,
            "std_sample": desc.get("std_sample"),
            "variance_population": desc.get("variance_population"),
            "variance_sample": desc.get("variance_sample"),
            "iqr": None,
        },
        "quantiles": desc.get("quantiles", {}),
        "shape": {
            "skew": skew_val,
            "kurtosis": kurt_val,
            **shape_summary,
        },
        "normality": sw,
        "distribution_fitting": {
            "top_k": top_k,
            "best_fit": best_dist,
            "ranked_distributions": top_dists,
        },
    }

    q = desc.get("quantiles", {})
    p75 = q.get("P75")
    p25 = q.get("P25")
    if p75 is not None and p25 is not None:
        report_cards["dispersion"]["iqr"] = float(p75 - p25)

    return {
        "generated_at": datetime.now().isoformat(),
        "column_name": col_name,
        "unit": unit,
        "alpha": alpha,
        "cards": report_cards,
        "warnings": warnings,
        "raw_sections": {
            "descriptive": desc,
            "normality_test": sw,
            "auto_fit": fit_result,
        },
    }
