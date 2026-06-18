import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any
from datetime import datetime


@dataclass
class DataSet:
    data: Union[List[float], List[List[float]], np.ndarray]
    column_names: Optional[List[str]] = None
    units: Optional[Dict[str, str]] = None
    missing_value_marker: Any = np.nan
    name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self._is_matrix = isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], list)
        if self._is_matrix:
            n_outer = len(self.data)
            n_inner = len(self.data[0]) if n_outer > 0 else 0
            if self.column_names is not None and len(self.column_names) == n_outer and len(self.column_names) != n_inner:
                self.data = np.array(self.data, dtype=object).T
                self.n_rows = n_inner
                self.n_cols = n_outer
            else:
                self.data = np.array(self.data, dtype=object)
                self.n_rows = n_outer
                self.n_cols = n_inner
        else:
            self.data = np.asarray(self.data, dtype=float)
            self.n_rows = len(self.data)
            self.n_cols = 1
        if self.column_names is None:
            self.column_names = [f"col_{i}" for i in range(self.n_cols)] if self._is_matrix else ["value"]
        if self.units is None:
            self.units = {}

    @staticmethod
    def _safe_float(val):
        try:
            f = float(val)
            return f if not np.isnan(f) or (isinstance(val, float) and np.isnan(val)) else f
        except (ValueError, TypeError):
            return np.nan

    def get_raw_column(self, col_idx: int = 0) -> np.ndarray:
        if self._is_matrix:
            col = self.data[:, col_idx]
        else:
            col = self.data
        return np.array([self._safe_float(v) for v in col], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": [self.n_rows, self.n_cols] if self._is_matrix else [self.n_rows],
            "column_names": self.column_names,
            "units": self.units,
            "missing_value_marker": str(self.missing_value_marker),
            "created_at": self.created_at,
            "is_matrix": self._is_matrix,
            "total_values": int(self.data.size),
            "missing_count": self.get_nan_count(),
        }

    def _is_missing(self, val) -> bool:
        try:
            return val is None or (isinstance(val, float) and np.isnan(val))
        except:
            return False

    def get_clean_column(self, col_idx: int = 0) -> np.ndarray:
        raw = self.get_raw_column(col_idx)
        mask = ~np.isnan(raw)
        return raw[mask]

    def get_nan_count(self, col_idx: Optional[int] = None) -> int:
        if col_idx is not None:
            raw = self.get_raw_column(col_idx)
            return int(np.isnan(raw).sum())
        else:
            return int(sum(self.get_nan_count(i) for i in range(self.n_cols)))


@dataclass
class Distribution:
    name: str
    parameters: Dict[str, float]
    support: Optional[str] = None
    description: Optional[str] = None
    fitted: bool = False
    goodness_of_fit: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "parameters": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in self.parameters.items()},
            "fitted": self.fitted,
            "support": self.support,
            "description": self.description,
        }
        if self.goodness_of_fit is not None:
            result["goodness_of_fit"] = self.goodness_of_fit
        return result


@dataclass
class TestResult:
    test_name: str
    test_type: str
    input_params: Dict[str, Any]
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    degrees_of_freedom: Optional[Union[float, List[float]]] = None
    significance_level: float = 0.05
    conclusion: Optional[str] = None
    extra_info: Optional[Dict[str, Any]] = None
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.conclusion is None and self.p_value is not None:
            self.conclusion = (
                f"Reject H0 (p={self.p_value:.4f} < {self.significance_level})"
                if self.p_value < self.significance_level
                else f"Fail to reject H0 (p={self.p_value:.4f} >= {self.significance_level})"
            )

    def to_dict(self) -> Dict[str, Any]:
        def _convert(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v
        result = {
            "test_name": self.test_name,
            "test_type": self.test_type,
            "input_params": {k: _convert(v) for k, v in self.input_params.items()},
            "statistic": float(self.statistic) if self.statistic is not None else None,
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "degrees_of_freedom": self.degrees_of_freedom if isinstance(self.degrees_of_freedom, list) else (float(self.degrees_of_freedom) if self.degrees_of_freedom is not None else None),
            "significance_level": self.significance_level,
            "conclusion": self.conclusion,
            "executed_at": self.executed_at,
        }
        if self.extra_info is not None:
            result["extra_info"] = {k: _convert(v) for k, v in self.extra_info.items()}
        return result


@dataclass
class RegressionModel:
    model_type: str
    parameters: Dict[str, Any]
    features: Optional[List[str]] = None
    target: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    fitted: bool = False
    model_id: str = field(default_factory=lambda: f"model_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    trained_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _convert(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "fitted": self.fitted,
            "features": self.features,
            "target": self.target,
            "parameters": {k: _convert(v) for k, v in self.parameters.items()},
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else _convert(v) for k, v in (self.metrics or {}).items()},
            "hyperparameters": {k: _convert(v) for k, v in (self.hyperparameters or {}).items()},
            "trained_at": self.trained_at,
            "metadata": self.metadata,
        }
