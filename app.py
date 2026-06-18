import os
import json
import traceback
from flask import Flask, request, jsonify

from models import DataSet
import statistics_core as sc
import distributions as dist
import hypothesis_tests as ht
import regression as reg
from report import full_statistics_report

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

_LOADED_MODELS: dict = {}


def _json_response(data, status=200):
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False, default=str),
        status=status,
        mimetype="application/json",
    )


def _error_response(msg, status=400, details=None):
    payload = {"error": msg}
    if details:
        payload["details"] = details
    return _json_response(payload, status)


@app.route("/", methods=["GET"])
def index():
    return _json_response({
        "service": "Statistics Library Backend Service",
        "version": "1.0.0",
        "port": 5003,
        "status": "running",
        "endpoints": {
            "datasets": [
                "POST /api/datasets/create",
                "POST /api/datasets/info",
            ],
            "descriptive_statistics": [
                "POST /api/stats/univariate",
                "POST /api/stats/correlation",
                "POST /api/stats/covariance",
                "POST /api/stats/summary",
            ],
            "distributions": [
                "GET  /api/distributions/catalog",
                "POST /api/distributions/fit",
                "POST /api/distributions/auto-fit",
                "POST /api/distributions/pdf",
                "POST /api/distributions/cdf",
                "POST /api/distributions/ppf",
                "POST /api/distributions/sf",
                "POST /api/distributions/isf",
                "POST /api/distributions/rvs",
            ],
            "hypothesis_tests": [
                "POST /api/tests/t/one-sample",
                "POST /api/tests/t/two-independent",
                "POST /api/tests/t/paired",
                "POST /api/tests/anova",
                "POST /api/tests/chi2/gof",
                "POST /api/tests/chi2/independence",
                "POST /api/tests/wilcoxon",
                "POST /api/tests/mann-whitney",
                "POST /api/tests/kruskal",
                "POST /api/tests/shapiro",
                "POST /api/tests/correlation",
            ],
            "regression": [
                "POST /api/regression/fit/linear",
                "POST /api/regression/fit/polynomial",
                "POST /api/regression/fit/ridge",
                "POST /api/regression/fit/lasso",
                "POST /api/regression/fit/logistic",
                "POST /api/regression/predict",
                "POST /api/regression/score",
                "POST /api/regression/save",
                "POST /api/regression/load",
                "GET  /api/regression/models",
                "DELETE /api/regression/models/<filename>",
            ],
            "report": [
                "POST /api/report/full",
            ],
        },
    })


@app.route("/api/health", methods=["GET"])
def health():
    return _json_response({"status": "ok"})


@app.route("/api/datasets/create", methods=["POST"])
def create_dataset():
    try:
        body = request.get_json(force=True)
        ds = DataSet(
            data=body["data"],
            column_names=body.get("column_names"),
            units=body.get("units"),
            missing_value_marker=body.get("missing_value_marker", float("nan")),
            name=body.get("name"),
        )
        return _json_response({"dataset": ds.to_dict()})
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/stats/univariate", methods=["POST"])
def univariate():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
            col_idx = body.get("column_idx", 0)
        else:
            col_idx = body.get("column_idx", 0)
            ds = data
        result = sc.descriptive_univariate(
            ds,
            ddof=body.get("ddof", 0),
            quantiles=body.get("quantiles"),
            column_idx=col_idx,
        )
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/stats/correlation", methods=["POST"])
def correlation():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
        else:
            ds = data
        method = body.get("method", "pearson")
        return _json_response(sc.correlation_matrix(ds, method=method))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/stats/covariance", methods=["POST"])
def covariance():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
        else:
            ds = data
        return _json_response(sc.covariance_matrix(ds, ddof=body.get("ddof", 1)))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/stats/summary", methods=["POST"])
def summary():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
        else:
            ds = data
        return _json_response(sc.summary_stats(ds, ddof=body.get("ddof", 0), quantiles=body.get("quantiles")))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/distributions/catalog", methods=["GET"])
def dist_catalog():
    try:
        return _json_response(dist.list_distributions())
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/distributions/fit", methods=["POST"])
def dist_fit():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        dist_name = body["distribution"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
            col_idx = body.get("column_idx", 0)
        else:
            ds = data
            col_idx = body.get("column_idx", 0)
        fitted = dist.fit_distribution(ds, dist_name, column_idx=col_idx)
        return _json_response(fitted.to_dict())
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/distributions/auto-fit", methods=["POST"])
def dist_auto_fit():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
            col_idx = body.get("column_idx", 0)
        else:
            ds = data
            col_idx = body.get("column_idx", 0)
        result = dist.auto_fit_distributions(
            ds,
            candidates=body.get("candidates"),
            top_k=body.get("top_k", 3),
            column_idx=col_idx,
            rank_by=body.get("rank_by", "ad_statistic"),
        )
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


def _dist_query(fn):
    def _handler():
        try:
            body = request.get_json(force=True)
            name = body["distribution"]
            params = body.get("parameters")
            var_key = [k for k in body if k in ("x", "q", "size")][0]
            value = body[var_key]
            kwargs = {"dist_name": name, "parameters": params}
            if var_key == "size":
                kwargs["size"] = value
                kwargs["random_state"] = body.get("random_state")
            else:
                kwargs[var_key] = value
            return _json_response(fn(**kwargs))
        except Exception as e:
            return _error_response(str(e), 400, traceback.format_exc())
    _handler.__name__ = f"dist_{fn.__name__}"
    return _handler


app.add_url_rule("/api/distributions/pdf", view_func=_dist_query(dist.distribution_pdf), methods=["POST"])
app.add_url_rule("/api/distributions/cdf", view_func=_dist_query(dist.distribution_cdf), methods=["POST"])
app.add_url_rule("/api/distributions/ppf", view_func=_dist_query(dist.distribution_ppf), methods=["POST"])
app.add_url_rule("/api/distributions/sf", view_func=_dist_query(dist.distribution_sf), methods=["POST"])
app.add_url_rule("/api/distributions/isf", view_func=_dist_query(dist.distribution_isf), methods=["POST"])
app.add_url_rule("/api/distributions/rvs", view_func=_dist_query(dist.distribution_rvs), methods=["POST"])


def _extract_sample(body, key):
    v = body[key]
    if isinstance(v, dict) and "data" in v:
        col = v.get("column_idx", 0)
        ds = DataSet(v["data"], v.get("column_names"))
        return ds, col
    return v, None


@app.route("/api/tests/t/one-sample", methods=["POST"])
def t_one_sample():
    try:
        body = request.get_json(force=True)
        sample, col = _extract_sample(body, "data")
        return _json_response(ht.ttest_one_sample(
            sample,
            popmean=body["popmean"],
            alternative=body.get("alternative", "two-sided"),
            alpha=body.get("alpha", 0.05),
            column_idx=col if col is not None else body.get("column_idx"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/t/two-independent", methods=["POST"])
def t_two_independent():
    try:
        body = request.get_json(force=True)
        s1, c1 = _extract_sample(body, "sample1")
        s2, c2 = _extract_sample(body, "sample2")
        return _json_response(ht.ttest_two_independent(
            s1, s2,
            equal_var=body.get("equal_var", False),
            alternative=body.get("alternative", "two-sided"),
            alpha=body.get("alpha", 0.05),
            col1=c1 if c1 is not None else body.get("col1"),
            col2=c2 if c2 is not None else body.get("col2"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/t/paired", methods=["POST"])
def t_paired():
    try:
        body = request.get_json(force=True)
        s1, c1 = _extract_sample(body, "sample1")
        s2, c2 = _extract_sample(body, "sample2")
        return _json_response(ht.ttest_paired(
            s1, s2,
            alternative=body.get("alternative", "two-sided"),
            alpha=body.get("alpha", 0.05),
            col1=c1 if c1 is not None else body.get("col1"),
            col2=c2 if c2 is not None else body.get("col2"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/anova", methods=["POST"])
def anova_route():
    try:
        body = request.get_json(force=True)
        groups = body["groups"]
        return _json_response(ht.anova(*groups, alpha=body.get("alpha", 0.05)))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/chi2/gof", methods=["POST"])
def chi2_gof():
    try:
        body = request.get_json(force=True)
        return _json_response(ht.chi2_goodness_of_fit(
            observed=body["observed"],
            expected=body.get("expected"),
            alpha=body.get("alpha", 0.05),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/chi2/independence", methods=["POST"])
def chi2_indep():
    try:
        body = request.get_json(force=True)
        return _json_response(ht.chi2_independence(
            contingency=body["contingency"],
            alpha=body.get("alpha", 0.05),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/wilcoxon", methods=["POST"])
def wilcoxon():
    try:
        body = request.get_json(force=True)
        s1, c1 = _extract_sample(body, "sample1")
        s2, c2 = _extract_sample(body, "sample2")
        return _json_response(ht.wilcoxon_signed_rank(
            s1, s2,
            alternative=body.get("alternative", "two-sided"),
            alpha=body.get("alpha", 0.05),
            col1=c1 if c1 is not None else body.get("col1"),
            col2=c2 if c2 is not None else body.get("col2"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/mann-whitney", methods=["POST"])
def mann_whitney():
    try:
        body = request.get_json(force=True)
        s1, c1 = _extract_sample(body, "sample1")
        s2, c2 = _extract_sample(body, "sample2")
        return _json_response(ht.mann_whitney_u(
            s1, s2,
            alternative=body.get("alternative", "two-sided"),
            alpha=body.get("alpha", 0.05),
            col1=c1 if c1 is not None else body.get("col1"),
            col2=c2 if c2 is not None else body.get("col2"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/kruskal", methods=["POST"])
def kruskal():
    try:
        body = request.get_json(force=True)
        groups = body["groups"]
        return _json_response(ht.kruskal_wallis(*groups, alpha=body.get("alpha", 0.05)))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/shapiro", methods=["POST"])
def shapiro():
    try:
        body = request.get_json(force=True)
        sample, col = _extract_sample(body, "data")
        return _json_response(ht.shapiro_wilk(
            sample,
            alpha=body.get("alpha", 0.05),
            column_idx=col if col is not None else body.get("column_idx"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/tests/correlation", methods=["POST"])
def corr_test():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"))
        else:
            ds = data
        return _json_response(ht.correlation_test(
            ds,
            method=body.get("method", "pearson"),
            alpha=body.get("alpha", 0.05),
            col_indices=body.get("col_indices"),
        ))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


def _extract_X(body):
    d = body["X"]
    if isinstance(d, dict) and "data" in d:
        ds = DataSet(d["data"], d.get("column_names"))
        return ds, d.get("column_names")
    return d, body.get("feature_names")


def _do_fit(fit_fn, body, **extra):
    try:
        X, fn = _extract_X(body)
        y = body["y"]
        fit_result = fit_fn(
            X, y,
            feature_names=fn,
            target_name=body.get("target_name"),
            **extra,
        )
        mi = fit_result["model"].model_id
        _LOADED_MODELS[mi] = {"sklearn_model": fit_result["sklearn_model"], "model_info": fit_result["model"]}
        result = {"info": fit_result["info"], "model_id": mi, "state": "loaded_in_memory"}
        if body.get("save", False):
            sv = reg.save_model(fit_result["sklearn_model"], fit_result["model"], filepath=body.get("save_path"))
            result["saved"] = sv
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/fit/linear", methods=["POST"])
def fit_lin():
    body = request.get_json(force=True)
    return _do_fit(reg.fit_linear, body, fit_intercept=body.get("fit_intercept", True))


@app.route("/api/regression/fit/polynomial", methods=["POST"])
def fit_poly():
    body = request.get_json(force=True)
    return _do_fit(reg.fit_polynomial, body,
                   degree=body.get("degree", 2),
                   interaction_only=body.get("interaction_only", False),
                   fit_intercept=body.get("fit_intercept", True))


@app.route("/api/regression/fit/ridge", methods=["POST"])
def fit_ridge():
    body = request.get_json(force=True)
    return _do_fit(reg.fit_ridge, body,
                   alpha=body.get("alpha", 1.0),
                   fit_intercept=body.get("fit_intercept", True),
                   normalize=body.get("normalize", False))


@app.route("/api/regression/fit/lasso", methods=["POST"])
def fit_lasso():
    body = request.get_json(force=True)
    return _do_fit(reg.fit_lasso, body,
                   alpha=body.get("alpha", 1.0),
                   fit_intercept=body.get("fit_intercept", True),
                   normalize=body.get("normalize", False),
                   max_iter=body.get("max_iter", 10000))


@app.route("/api/regression/fit/logistic", methods=["POST"])
def fit_logistic():
    body = request.get_json(force=True)
    return _do_fit(reg.fit_logistic, body,
                   C=body.get("C", 1.0),
                   penalty=body.get("penalty", "l2"),
                   solver=body.get("solver", "lbfgs"),
                   max_iter=body.get("max_iter", 1000),
                   fit_intercept=body.get("fit_intercept", True))


def _resolve_model(body):
    mid = body.get("model_id")
    path = body.get("filepath")
    if mid and mid in _LOADED_MODELS:
        return _LOADED_MODELS[mid]["sklearn_model"], mid
    if path:
        ld = reg.load_model(path)
        skm = ld["sklearn_model"]
        info = ld["model_info"]
        _LOADED_MODELS[info.model_id] = {"sklearn_model": skm, "model_info": info}
        return skm, info.model_id
    raise ValueError("Model not found: specify model_id (loaded) or filepath")


@app.route("/api/regression/predict", methods=["POST"])
def reg_predict():
    try:
        body = request.get_json(force=True)
        skm, mid = _resolve_model(body)
        Xd = body["X"]
        if isinstance(Xd, dict) and "data" in Xd:
            ds = DataSet(Xd["data"], Xd.get("column_names"))
            X = ds
        else:
            X = Xd
        result = reg.predict_model(skm, X)
        result["model_id"] = mid
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/score", methods=["POST"])
def reg_score():
    try:
        body = request.get_json(force=True)
        skm, mid = _resolve_model(body)
        Xd = body["X"]
        if isinstance(Xd, dict) and "data" in Xd:
            X = DataSet(Xd["data"], Xd.get("column_names"))
        else:
            X = Xd
        result = reg.score_model(skm, X, body["y"])
        result["model_id"] = mid
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/save", methods=["POST"])
def reg_save():
    try:
        body = request.get_json(force=True)
        mid = body["model_id"]
        if mid not in _LOADED_MODELS:
            return _error_response(f"Model {mid} not loaded", 404)
        skm = _LOADED_MODELS[mid]["sklearn_model"]
        info = _LOADED_MODELS[mid]["model_info"]
        result = reg.save_model(skm, info, filepath=body.get("filepath"))
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/load", methods=["POST"])
def reg_load():
    try:
        body = request.get_json(force=True)
        ld = reg.load_model(body["filepath"])
        mid = ld["model_info"].model_id
        _LOADED_MODELS[mid] = {"sklearn_model": ld["sklearn_model"], "model_info": ld["model_info"]}
        return _json_response({
            "loaded": True,
            "model_id": mid,
            "info": ld["info"],
            "filepath": ld["filepath"],
            "saved_at": ld.get("saved_at"),
            "state": "loaded_in_memory",
        })
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/models", methods=["GET"])
def reg_list():
    try:
        disk = reg.list_saved_models()
        memory = [
            {
                "model_id": mid,
                "model_type": info["model_info"].model_type,
                "fitted": info["model_info"].fitted,
            }
            for mid, info in _LOADED_MODELS.items()
        ]
        return _json_response({"disk": disk, "memory": memory})
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/regression/models/<path:filename>", methods=["DELETE"])
def reg_delete(filename):
    try:
        return _json_response(reg.delete_model(filename))
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


@app.route("/api/report/full", methods=["POST"])
def report_full():
    try:
        body = request.get_json(force=True)
        data = body["data"]
        ds = None
        if isinstance(data, dict) and "data" in data:
            ds = DataSet(data["data"], data.get("column_names"), units=data.get("units"))
        else:
            ds = data
        result = full_statistics_report(
            ds,
            column_idx=body.get("column_idx", 0),
            quantiles=body.get("quantiles"),
            top_k=body.get("top_k", 3),
            fit_candidates=body.get("fit_candidates"),
            alpha=body.get("alpha", 0.05),
        )
        return _json_response(result)
    except Exception as e:
        return _error_response(str(e), 400, traceback.format_exc())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    print(f"Starting Statistics Service on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
