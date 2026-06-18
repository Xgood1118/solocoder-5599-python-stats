import numpy as np
from typing import List, Dict, Any, Optional, Union
from scipy import stats
from models import TestResult, DataSet


def _clean_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mask = ~np.isnan(arr)
    return arr[mask]


def _skipna_infos(arrays: List[np.ndarray]) -> List[Dict[str, int]]:
    infos = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        total = len(a)
        clean = _clean_array(a)
        infos.append({"total_count": total, "valid_count": len(clean), "skipped_nan": total - len(clean)})
    return infos


def _paired_skipna(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return {
        "x": x[mask],
        "y": y[mask],
        "skipna": {
            "total_count": int(len(x)),
            "valid_pairs": int(mask.sum()),
            "skipped_nan_rows": int(len(x) - mask.sum()),
        },
    }


def ttest_one_sample(data: Union[List[float], np.ndarray, DataSet],
                     popmean: float,
                     alternative: str = "two-sided",
                     alpha: float = 0.05,
                     column_idx: Optional[int] = None) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        arr = data.get_clean_column(col_idx)
        skipna = [{"total_count": data.n_rows, "valid_count": len(arr), "skipped_nan": data.n_rows - len(arr)}]
    else:
        arr_raw = np.asarray(data, dtype=float)
        arr = _clean_array(arr_raw)
        skipna = _skipna_infos([arr_raw])

    stat, p_value = stats.ttest_1samp(arr, popmean=popmean, alternative=alternative, nan_policy="omit")
    result = TestResult(
        test_name="One-Sample t-test",
        test_type="parametric",
        input_params={"popmean": popmean, "alternative": alternative, "alpha": alpha, "n": len(arr)},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=float(len(arr) - 1),
        significance_level=alpha,
        extra_info={"skipna_per_sample": skipna, "sample_mean": float(np.mean(arr)), "sample_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else None},
    )
    return {"test": result.to_dict(), "h0": f"mean = {popmean}", "h1": f"mean {alternative} {popmean}"}


def ttest_two_independent(sample1: Union[List[float], np.ndarray, DataSet],
                          sample2: Union[List[float], np.ndarray, DataSet],
                          equal_var: bool = False,
                          alternative: str = "two-sided",
                          alpha: float = 0.05,
                          col1: Optional[int] = None,
                          col2: Optional[int] = None) -> Dict[str, Any]:
    def _extract(d, col):
        if isinstance(d, DataSet):
            ci = col if col is not None else 0
            arr = d.get_clean_column(ci)
            info = {"total_count": d.n_rows, "valid_count": len(arr), "skipped_nan": d.n_rows - len(arr)}
            return arr, info
        else:
            raw = np.asarray(d, dtype=float)
            arr = _clean_array(raw)
            info = {"total_count": len(raw), "valid_count": len(arr), "skipped_nan": len(raw) - len(arr)}
            return arr, info

    a, info1 = _extract(sample1, col1)
    b, info2 = _extract(sample2, col2)

    if equal_var:
        test_name = "Two-Sample t-test (equal variance, Student)"
    else:
        test_name = "Welch's t-test (unequal variance)"
    stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative, nan_policy="omit")

    def _welch_df(x, y):
        nx, ny = len(x), len(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        return float((vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))) if (vx > 0 and vy > 0 and nx > 1 and ny > 1) else None

    df_val = float(len(a) + len(b) - 2) if equal_var else _welch_df(a, b)

    result = TestResult(
        test_name=test_name,
        test_type="parametric",
        input_params={"equal_var": equal_var, "alternative": alternative, "alpha": alpha, "n1": len(a), "n2": len(b)},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=df_val,
        significance_level=alpha,
        extra_info={
            "skipna_per_sample": [info1, info2],
            "mean1": float(np.mean(a)),
            "mean2": float(np.mean(b)),
            "var1": float(np.var(a, ddof=1)) if len(a) > 1 else None,
            "var2": float(np.var(b, ddof=1)) if len(b) > 1 else None,
        },
    )
    alt_map = {"two-sided": "!=", "greater": ">", "less": "<"}
    return {"test": result.to_dict(), "h0": "mean1 = mean2", "h1": f"mean1 {alt_map.get(alternative, alternative)} mean2"}


def ttest_paired(sample1: Union[List[float], np.ndarray, DataSet],
                 sample2: Union[List[float], np.ndarray, DataSet],
                 alternative: str = "two-sided",
                 alpha: float = 0.05,
                 col1: Optional[int] = None,
                 col2: Optional[int] = None) -> Dict[str, Any]:
    def _extract(d, col):
        if isinstance(d, DataSet):
            ci = col if col is not None else 0
            return d.data[:, ci].astype(float, errors="ignore") if d._is_matrix else d.data.astype(float, errors="ignore")
        return np.asarray(d, dtype=float)

    x_raw = _extract(sample1, col1)
    y_raw = _extract(sample2, col2)
    if len(x_raw) != len(y_raw):
        raise ValueError("Paired samples must have equal length")

    paired = _paired_skipna(x_raw, y_raw)
    x, y = paired["x"], paired["y"]
    stat, p_value = stats.ttest_rel(x, y, alternative=alternative, nan_policy="omit")

    diff = x - y
    result = TestResult(
        test_name="Paired t-test (Dependent samples)",
        test_type="parametric",
        input_params={"alternative": alternative, "alpha": alpha, "n_pairs": len(x)},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=float(len(x) - 1),
        significance_level=alpha,
        extra_info={
            "skipna": paired["skipna"],
            "mean_difference": float(np.mean(diff)),
            "std_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else None,
            "mean1": float(np.mean(x)),
            "mean2": float(np.mean(y)),
        },
    )
    alt_map = {"two-sided": "!=", "greater": ">", "less": "<"}
    return {"test": result.to_dict(), "h0": "mean(diff) = 0 (mean1 = mean2)", "h1": f"mean(diff) {alt_map.get(alternative, alternative)} 0"}


def anova(*groups: Union[List[float], np.ndarray],
          alpha: float = 0.05) -> Dict[str, Any]:
    cleaned = []
    skipna_list = []
    for g in groups:
        g_raw = np.asarray(g, dtype=float)
        gc = _clean_array(g_raw)
        cleaned.append(gc)
        skipna_list.append({"total_count": len(g_raw), "valid_count": len(gc), "skipped_nan": len(g_raw) - len(gc)})

    stat, p_value = stats.f_oneway(*cleaned)
    k = len(cleaned)
    n_total = sum(len(g) for g in cleaned)
    result = TestResult(
        test_name="One-Way ANOVA (F-test)",
        test_type="parametric",
        input_params={"k_groups": k, "alpha": alpha, "group_sizes": [len(g) for g in cleaned]},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=[float(k - 1), float(n_total - k)],
        significance_level=alpha,
        extra_info={
            "skipna_per_group": skipna_list,
            "group_means": [float(np.mean(g)) for g in cleaned],
            "group_stds": [float(np.std(g, ddof=1)) if len(g) > 1 else None for g in cleaned],
        },
    )
    return {"test": result.to_dict(), "h0": "All group means are equal", "h1": "At least one group mean differs"}


def chi2_goodness_of_fit(observed: Union[List[int], List[float], np.ndarray],
                         expected: Optional[Union[List[float], np.ndarray]] = None,
                         alpha: float = 0.05) -> Dict[str, Any]:
    obs = np.asarray(observed, dtype=float)
    if expected is None:
        exp = np.full_like(obs, obs.sum() / len(obs))
    else:
        exp = np.asarray(expected, dtype=float)
        if exp.sum() != obs.sum():
            exp = exp * (obs.sum() / exp.sum())

    stat, p_value = stats.chisquare(obs, f_exp=exp)
    result = TestResult(
        test_name="Chi-Squared Goodness-of-Fit Test (χ²)",
        test_type="parametric",
        input_params={"k_categories": len(obs), "alpha": alpha, "observed_sum": float(obs.sum())},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=float(len(obs) - 1),
        significance_level=alpha,
        extra_info={
            "observed_frequencies": [float(x) for x in obs],
            "expected_frequencies": [float(x) for x in exp],
            "residuals": [float(o - e) for o, e in zip(obs, exp)],
        },
    )
    return {"test": result.to_dict(), "h0": "Observed distribution matches expected", "h1": "Observed distribution differs from expected"}


def chi2_independence(contingency: Union[List[List[int]], List[List[float]], np.ndarray],
                      alpha: float = 0.05) -> Dict[str, Any]:
    table = np.asarray(contingency, dtype=float)
    chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=True)
    result = TestResult(
        test_name="Chi-Squared Test for Independence (χ²)",
        test_type="parametric",
        input_params={"shape": [int(table.shape[0]), int(table.shape[1])], "alpha": alpha, "total_observations": float(table.sum())},
        statistic=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=float(dof),
        significance_level=alpha,
        extra_info={
            "observed": [[float(x) for x in row] for row in table],
            "expected": [[float(x) for x in row] for row in expected],
        },
    )
    return {"test": result.to_dict(), "h0": "The two categorical variables are independent", "h1": "The two categorical variables are associated"}


def wilcoxon_signed_rank(sample1: Union[List[float], np.ndarray, DataSet],
                         sample2: Union[List[float], np.ndarray, DataSet],
                         alternative: str = "two-sided",
                         alpha: float = 0.05,
                         col1: Optional[int] = None,
                         col2: Optional[int] = None) -> Dict[str, Any]:
    def _extract(d, col):
        if isinstance(d, DataSet):
            ci = col if col is not None else 0
            return d.data[:, ci].astype(float, errors="ignore") if d._is_matrix else d.data.astype(float, errors="ignore")
        return np.asarray(d, dtype=float)

    x_raw = _extract(sample1, col1)
    y_raw = _extract(sample2, col2)
    if len(x_raw) != len(y_raw):
        raise ValueError("Paired samples must have equal length")

    paired = _paired_skipna(x_raw, y_raw)
    x, y = paired["x"], paired["y"]
    res = stats.wilcoxon(x, y, alternative=alternative, mode="auto")
    stat = float(res.statistic)
    p_value = float(res.pvalue)

    result = TestResult(
        test_name="Wilcoxon Signed-Rank Test (Non-parametric paired)",
        test_type="non_parametric",
        input_params={"alternative": alternative, "alpha": alpha, "n_pairs": len(x)},
        statistic=stat,
        p_value=p_value,
        degrees_of_freedom=None,
        significance_level=alpha,
        extra_info={"skipna": paired["skipna"], "median_difference": float(np.median(x - y))},
    )
    alt_map = {"two-sided": "!=", "greater": ">", "less": "<"}
    return {"test": result.to_dict(), "h0": "Median of differences = 0", "h1": f"Median(diff) {alt_map.get(alternative, alternative)} 0"}


def mann_whitney_u(sample1: Union[List[float], np.ndarray, DataSet],
                   sample2: Union[List[float], np.ndarray, DataSet],
                   alternative: str = "two-sided",
                   alpha: float = 0.05,
                   col1: Optional[int] = None,
                   col2: Optional[int] = None) -> Dict[str, Any]:
    def _extract(d, col):
        if isinstance(d, DataSet):
            ci = col if col is not None else 0
            arr = d.get_clean_column(ci)
            info = {"total_count": d.n_rows, "valid_count": len(arr), "skipped_nan": d.n_rows - len(arr)}
            return arr, info
        else:
            raw = np.asarray(d, dtype=float)
            arr = _clean_array(raw)
            info = {"total_count": len(raw), "valid_count": len(arr), "skipped_nan": len(raw) - len(arr)}
            return arr, info

    a, info1 = _extract(sample1, col1)
    b, info2 = _extract(sample2, col2)
    stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)

    result = TestResult(
        test_name="Mann-Whitney U Test (Non-parametric two-sample)",
        test_type="non_parametric",
        input_params={"alternative": alternative, "alpha": alpha, "n1": len(a), "n2": len(b)},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=None,
        significance_level=alpha,
        extra_info={
            "skipna_per_sample": [info1, info2],
            "median1": float(np.median(a)),
            "median2": float(np.median(b)),
        },
    )
    alt_map = {"two-sided": "!=", "greater": ">", "less": "<"}
    return {"test": result.to_dict(), "h0": "Two samples come from same distribution", "h1": f"Distributions differ (stochastically {alt_map.get(alternative, alternative)})"}


def kruskal_wallis(*groups: Union[List[float], np.ndarray],
                   alpha: float = 0.05) -> Dict[str, Any]:
    cleaned = []
    skipna_list = []
    for g in groups:
        g_raw = np.asarray(g, dtype=float)
        gc = _clean_array(g_raw)
        cleaned.append(gc)
        skipna_list.append({"total_count": len(g_raw), "valid_count": len(gc), "skipped_nan": len(g_raw) - len(gc)})

    stat, p_value = stats.kruskal(*cleaned)
    result = TestResult(
        test_name="Kruskal-Wallis H Test (Non-parametric k-sample ANOVA)",
        test_type="non_parametric",
        input_params={"k_groups": len(cleaned), "alpha": alpha, "group_sizes": [len(g) for g in cleaned]},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=float(len(cleaned) - 1),
        significance_level=alpha,
        extra_info={
            "skipna_per_group": skipna_list,
            "group_medians": [float(np.median(g)) for g in cleaned],
        },
    )
    return {"test": result.to_dict(), "h0": "All groups have same median", "h1": "At least one group median differs"}


def shapiro_wilk(data: Union[List[float], np.ndarray, DataSet],
                 alpha: float = 0.05,
                 column_idx: Optional[int] = None) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        col_idx = column_idx if column_idx is not None else 0
        arr = data.get_clean_column(col_idx)
        skipna = {"total_count": data.n_rows, "valid_count": len(arr), "skipped_nan": data.n_rows - len(arr)}
    else:
        arr_raw = np.asarray(data, dtype=float)
        arr = _clean_array(arr_raw)
        skipna = {"total_count": len(arr_raw), "valid_count": len(arr), "skipped_nan": len(arr_raw) - len(arr)}

    stat, p_value = stats.shapiro(arr)
    result = TestResult(
        test_name="Shapiro-Wilk Normality Test",
        test_type="normality",
        input_params={"alpha": alpha, "n": len(arr)},
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=None,
        significance_level=alpha,
        extra_info={"skipna": skipna, "note": "Reliable for n in [3, 5000]"},
    )
    return {"test": result.to_dict(), "h0": "Data is normally distributed", "h1": "Data is not normally distributed"}


def correlation_test(data: Union[List[List[float]], np.ndarray, DataSet],
                     method: str = "pearson",
                     alpha: float = 0.05,
                     col_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    if isinstance(data, DataSet):
        if col_indices is not None:
            raw_cols = [data.data[:, i].astype(float, errors="ignore") if data._is_matrix else data.data.astype(float, errors="ignore") for i in col_indices]
            col_names = [data.column_names[i] for i in col_indices]
        else:
            raw_cols = [data.data[:, i].astype(float, errors="ignore") if data._is_matrix else data.data.astype(float, errors="ignore") for i in range(data.n_cols)]
            col_names = data.column_names
    else:
        mat = np.asarray(data, dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        if col_indices is not None:
            raw_cols = [mat[:, i] for i in col_indices]
            col_names = [f"col_{i}" for i in col_indices]
        else:
            raw_cols = [mat[:, i] for i in range(mat.shape[1])]
            col_names = [f"col_{i}" for i in range(mat.shape[1])]

    n = len(raw_cols)
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{col_names[i]}__{col_names[j]}"
            mask = ~np.isnan(raw_cols[i]) & ~np.isnan(raw_cols[j])
            x = raw_cols[i][mask]
            y = raw_cols[j][mask]
            skipna = {"total_count": int(len(raw_cols[i])), "valid_pairs": int(mask.sum()), "skipped_nan_rows": int(len(raw_cols[i]) - mask.sum())}

            if len(x) >= 3:
                if method == "pearson":
                    r, p = stats.pearsonr(x, y)
                elif method == "spearman":
                    r, p = stats.spearmanr(x, y)
                elif method == "kendall":
                    r, p = stats.kendalltau(x, y)
                else:
                    raise ValueError(f"Unknown method: {method}")

                test_res = TestResult(
                    test_name=f"{method.title()} Correlation Test",
                    test_type="correlation",
                    input_params={"method": method, "alpha": alpha, "n_pairs": len(x), "columns": key},
                    statistic=float(r),
                    p_value=float(p),
                    degrees_of_freedom=float(len(x) - 2) if method == "pearson" else None,
                    significance_level=alpha,
                    extra_info={"skipna": skipna, "correlation": float(r)},
                )
                results[key] = {
                    "correlation": float(r),
                    "p_value": float(p),
                    "test": test_res.to_dict(),
                    "h0": f"No {method} correlation (ρ=0)",
                    "h1": f"Significant {method} correlation (ρ≠0)",
                }
            else:
                results[key] = {"correlation": None, "p_value": None, "note": "Not enough valid pairs (need >=3)"}
    return {"method": method, "column_names": col_names, "results": results}
