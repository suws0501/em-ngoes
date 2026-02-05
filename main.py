from __future__ import annotations

from pathlib import Path
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import shap

from utils import (
    tsfresh_rolling_features,
    get_full_dataset,
    embargo_split,
    run_rf_pipeline_from_split,
    run_hmm_rf_pipeline_from_split,
    strategy_backtesting_trailing_sl,
    strategy_backtesting_average_change_rate,
    equity_curve_comparison,
    performance_metrics,
    kalman_filter_price,
    set_debug,
    debug_block,
    set_plot_config,
    show_or_save_plot,
)


DATA_PATH = Path("data") / "data.parquet"
HIGHER_FREQUENCY_PATH = Path("data") / "higher_frequency_data.parquet"
TOGETHER_PATH = Path("data") / "together.parquet"

CONFIG = {
    "data_mode": "precomputed_features",
    "debug": True,
    "save_plots": True,
    "plots_dir": "plots",
    "event_threshold": 0.005,
    "embargo": {"test_proportion": 0.1, "period": 30},
    "rf": {"shap_sample_size": 200, "alpha": 0.4, "top_k_shap": 20},
    "rf_avg": {"shap_sample_size": 200, "alpha": 0.5, "top_k_shap": 20},
    "hmm": {
        "embargo_period": 30,
        "top_k_shap": 20,
        "shap_sample_size": 200,
        "alpha": 0.5,
        "n_components": 3,
        "hmm_iter": 600,
    },
    "hmm_avg": {
        "embargo_period": 30,
        "top_k_shap": 20,
        "shap_sample_size": 200,
        "alpha": 0.5,
        "n_components": 2,
        "hmm_iter": 600,
    },
    "training_iters": {
        "reducing_n_estimators": 2000,
        "base_n_estimators": 400,
        "meta_n_estimators": 200,
    },
    "early_stop": {
        "enabled": True,
        "val_frac": 0.2,
        "patience": 3,
        "min_delta": 1e-4,
        "step": 50,
    },
    "precision_gate": {
        "enabled": True,
        "target_precision": 0.6,
        "min_coverage": 0.05,
    },
    "run_examples": {
        "rf_trailing": True,
        "rf_avg": True,
        "hmm_trailing": True,
        "hmm_avg": True,
        "kalman": False,
        "metrics": True,
    },
}

RAW_TSFRESH_CONFIG = {
    "time_series_col": "close",
    "window_size": 2500,
    "start_idx": 2500,
    "preset": "comprehensive",
    "n_jobs": 0,
}


def load_parquet(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    return pd.read_parquet(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _get_first_series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if name not in df.columns:
        return None
    col = df[name]
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


def ensure_together_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Event" not in df.columns and "event" in df.columns:
        series = _get_first_series(df, "event")
        if series is not None:
            df["Event"] = series
    if "ATR" not in df.columns and "atr" in df.columns:
        series = _get_first_series(df, "atr")
        if series is not None:
            df["ATR"] = series
    if "ret_skew_20" not in df.columns:
        price_series = _get_first_series(df, "price")
        if price_series is None:
            raise KeyError("Missing 'price' column; cannot compute ret_skew_20.")
        returns = price_series.pct_change()
        df["ret_skew_20"] = returns.rolling(20).skew()
    return df


def build_together_dataset(data: pd.DataFrame, ts_config: dict, event_threshold: float) -> pd.DataFrame:
    debug_block(
        "build_together_dataset:start",
        rows=len(data),
        cols=len(data.columns),
        tsfresh_col=ts_config.get("time_series_col"),
    )
    time_series_features_df = tsfresh_rolling_features(
        data.copy(),
        time_series_col=ts_config["time_series_col"],
        window_size=ts_config["window_size"],
        start_idx=ts_config["start_idx"],
        preset=ts_config["preset"],
        n_jobs=ts_config["n_jobs"],
    )

    base = get_full_dataset(data.copy(), event_threshold)

    together = pd.merge(
        time_series_features_df,
        base,
        left_index=True,
        right_index=True,
        how="inner",
    )
    debug_block(
        "build_together_dataset:end",
        rows=len(together),
        cols=len(together.columns),
    )
    return together


def run_rf_trailing_example(together: pd.DataFrame, higher_frequency_data: pd.DataFrame, config: dict):
    debug_block(
        "run_rf_trailing_example:start",
        together_rows=len(together),
        hf_rows=len(higher_frequency_data),
    )
    df = together.copy()
    train_df, test_df = embargo_split(df.copy(), config["embargo"]["test_proportion"], config["embargo"]["period"])

    testset, reducing_feature_model, sample_X = run_rf_pipeline_from_split(
        train_df=train_df,
        test_df=test_df,
        shap_sample_size=config["rf"]["shap_sample_size"],
        alpha=config["rf"]["alpha"],
        top_k_shap=config["rf"]["top_k_shap"],
        reducing_n_estimators=config["training_iters"]["reducing_n_estimators"],
        base_n_estimators=config["training_iters"]["base_n_estimators"],
        meta_n_estimators=config["training_iters"]["meta_n_estimators"],
        early_stop=config["early_stop"] if config["early_stop"]["enabled"] else None,
        target_precision=(
            config["precision_gate"]["target_precision"]
            if config["precision_gate"]["enabled"]
            else None
        ),
        min_precision_coverage=config["precision_gate"]["min_coverage"],
    )

    explainer = shap.TreeExplainer(reducing_feature_model)
    shap_values = explainer.shap_values(sample_X)
    shap_values_class1 = shap_values[:, :, 1]

    shap.summary_plot(shap_values_class1, sample_X, plot_type="bar")
    show_or_save_plot("shap_summary_bar_rf_trailing")
    shap.summary_plot(shap_values_class1, sample_X)
    show_or_save_plot("shap_summary_rf_trailing")

    test_equity_curve, trade_log = strategy_backtesting_trailing_sl(
        testset["signals"],
        testset["price"],
        higher_frequency_data,
        testset["low"],
        testset["high"],
        testset["meta_label_probs"],
        atr=testset["ATR"],
        sl_k=3,
        tp_k=3,
    )

    strategy_equity_curve = test_equity_curve.copy()
    actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
    cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
    equity_curve_comparison(strategy_equity_curve, cumulative_equity_curve, "strategy", "actual")

    return testset, trade_log


def run_rf_avg_example(together: pd.DataFrame, config: dict):
    debug_block(
        "run_rf_avg_example:start",
        together_rows=len(together),
    )
    df = together.copy()
    train_df, test_df = embargo_split(df.copy(), config["embargo"]["test_proportion"], config["embargo"]["period"])

    testset, _, _ = run_rf_pipeline_from_split(
        train_df=train_df,
        test_df=test_df,
        shap_sample_size=config["rf_avg"]["shap_sample_size"],
        alpha=config["rf_avg"]["alpha"],
        top_k_shap=config["rf_avg"]["top_k_shap"],
        reducing_n_estimators=config["training_iters"]["reducing_n_estimators"],
        base_n_estimators=config["training_iters"]["base_n_estimators"],
        meta_n_estimators=config["training_iters"]["meta_n_estimators"],
        early_stop=config["early_stop"] if config["early_stop"]["enabled"] else None,
        target_precision=(
            config["precision_gate"]["target_precision"]
            if config["precision_gate"]["enabled"]
            else None
        ),
        min_precision_coverage=config["precision_gate"]["min_coverage"],
    )

    test_equity_curve, trade_log = strategy_backtesting_average_change_rate(
        predictions=testset["signals"],
        prices=testset["price"],
        meta_probs=testset["meta_label_probs"],
        train_set_price_event=train_df,
        prob_threshold=0.3,
    )

    strategy_equity_curve = test_equity_curve.copy()
    actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
    cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
    equity_curve_comparison(strategy_equity_curve, cumulative_equity_curve, "strategy", "actual")

    return testset, trade_log


def run_hmm_trailing_example(together: pd.DataFrame, higher_frequency_data: pd.DataFrame, config: dict):
    debug_block(
        "run_hmm_trailing_example:start",
        together_rows=len(together),
        hf_rows=len(higher_frequency_data),
    )
    df = together.copy()
    df["returns"] = df["price"].pct_change()
    df["log_ret2"] = np.log(df["returns"] ** 2 + 1e-8)
    df.dropna(subset=["returns"], inplace=True)

    train_df, test_df = embargo_split(df.copy(), config["embargo"]["test_proportion"], config["embargo"]["period"])

    art = run_hmm_rf_pipeline_from_split(
        train_df=train_df,
        test_df=test_df,
        embargo_period=config["hmm"]["embargo_period"],
        top_k_shap=config["hmm"]["top_k_shap"],
        shap_sample_size=config["hmm"]["shap_sample_size"],
        alpha=config["hmm"]["alpha"],
        n_components=config["hmm"]["n_components"],
        hmm_iter=config["hmm"]["hmm_iter"],
        reducing_n_estimators=config["training_iters"]["reducing_n_estimators"],
        base_n_estimators=config["training_iters"]["base_n_estimators"],
        meta_n_estimators=config["training_iters"]["meta_n_estimators"],
        early_stop=config["early_stop"] if config["early_stop"]["enabled"] else None,
        target_precision=(
            config["precision_gate"]["target_precision"]
            if config["precision_gate"]["enabled"]
            else None
        ),
        min_precision_coverage=config["precision_gate"]["min_coverage"],
    )

    testset = art["testset"]
    test_equity_curve, trade_log = strategy_backtesting_trailing_sl(
        testset["signals"],
        testset["price"],
        higher_frequency_data,
        testset["low"],
        testset["high"],
        testset["meta_label_probs"],
        atr=testset["ATR"],
        sl_k=3,
        tp_k=3,
    )

    strategy_equity_curve = test_equity_curve.copy()
    actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
    cumulative_equity_curve = (1 + actual_equity_curve).cumprod()

    equity_curve_comparison(strategy_equity_curve, cumulative_equity_curve, "strategy", "actual")
    return testset, trade_log


def run_hmm_avg_example(together: pd.DataFrame, config: dict):
    debug_block(
        "run_hmm_avg_example:start",
        together_rows=len(together),
    )
    df = together.copy()
    df["returns"] = df["price"].pct_change()
    df["log_ret2"] = np.log(df["returns"] ** 2 + 1e-8)
    df.dropna(subset=["returns"], inplace=True)

    train_df, test_df = embargo_split(df.copy(), config["embargo"]["test_proportion"], config["embargo"]["period"])

    art = run_hmm_rf_pipeline_from_split(
        train_df=train_df,
        test_df=test_df,
        embargo_period=config["hmm_avg"]["embargo_period"],
        top_k_shap=config["hmm_avg"]["top_k_shap"],
        shap_sample_size=config["hmm_avg"]["shap_sample_size"],
        alpha=config["hmm_avg"]["alpha"],
        n_components=config["hmm_avg"]["n_components"],
        hmm_iter=config["hmm_avg"]["hmm_iter"],
        reducing_n_estimators=config["training_iters"]["reducing_n_estimators"],
        base_n_estimators=config["training_iters"]["base_n_estimators"],
        meta_n_estimators=config["training_iters"]["meta_n_estimators"],
        early_stop=config["early_stop"] if config["early_stop"]["enabled"] else None,
        target_precision=(
            config["precision_gate"]["target_precision"]
            if config["precision_gate"]["enabled"]
            else None
        ),
        min_precision_coverage=config["precision_gate"]["min_coverage"],
    )

    testset = art["testset"]
    test_equity_curve, trade_log = strategy_backtesting_average_change_rate(
        predictions=testset["signals"],
        prices=testset["price"],
        meta_probs=testset["meta_label_probs"],
        train_set_price_event=train_df,
        prob_threshold=0.3,
    )

    strategy_equity_curve = test_equity_curve.copy()
    actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
    cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
    equity_curve_comparison(strategy_equity_curve, cumulative_equity_curve, "strategy", "actual")

    return testset, trade_log


def run_kalman_example(data: pd.DataFrame):
    debug_block(
        "run_kalman_example:start",
        rows=len(data),
        cols=len(data.columns),
    )
    if "price" in data.columns:
        series = data["price"]
    elif "close" in data.columns:
        series = data["close"]
    else:
        raise KeyError("Missing 'price' or 'close' column for Kalman filter.")
    xhat, phat, _ = kalman_filter_price(series, Q=1e-5, R=1e-2)
    debug_block(
        "run_kalman_example:end",
        rows=len(xhat),
    )
    return xhat, phat


def main() -> None:
    warnings.filterwarnings("ignore")

    set_debug(bool(CONFIG.get("debug", False)))
    set_plot_config(bool(CONFIG.get("save_plots", False)), CONFIG.get("plots_dir", "plots"))

    higher_frequency_data = normalize_columns(load_parquet(HIGHER_FREQUENCY_PATH, required=False))
    debug_block(
        "main:load_higher_frequency_data",
        loaded=higher_frequency_data is not None,
        rows=len(higher_frequency_data) if higher_frequency_data is not None else 0,
        cols=len(higher_frequency_data.columns) if higher_frequency_data is not None else 0,
    )

    if CONFIG["data_mode"] == "raw_ohlcv":
        data = normalize_columns(load_parquet(DATA_PATH))
        debug_block(
            "main:load_raw_data",
            rows=len(data),
            cols=len(data.columns),
        )
        required_raw_cols = {"close", "low", "high", "volume"}
        if not required_raw_cols.issubset(set(data.columns)):
            raise ValueError(
                "data_mode='raw_ohlcv' requires raw OHLCV columns: "
                f"{sorted(required_raw_cols)}. "
                f"Found: {sorted(data.columns)}"
            )
        together = build_together_dataset(data, RAW_TSFRESH_CONFIG, CONFIG["event_threshold"])
    elif CONFIG["data_mode"] == "precomputed_features":
        together = normalize_columns(load_parquet(TOGETHER_PATH))
        debug_block(
            "main:load_precomputed_together",
            rows=len(together),
            cols=len(together.columns),
        )
    else:
        raise ValueError("CONFIG['data_mode'] must be 'raw_ohlcv' or 'precomputed_features'.")
    together = ensure_together_features(together)
    debug_block(
        "main:ensure_together_features",
        rows=len(together),
        cols=len(together.columns),
        has_event="Event" in together.columns,
        has_atr="ATR" in together.columns,
        has_ret_skew="ret_skew_20" in together.columns,
    )

    if CONFIG["run_examples"]["rf_trailing"]:
        if higher_frequency_data is None:
            raise FileNotFoundError(
                f"Missing required file for trailing SL example: {HIGHER_FREQUENCY_PATH}"
            )
        run_rf_trailing_example(together, higher_frequency_data, CONFIG)

    if CONFIG["run_examples"]["rf_avg"]:
        run_rf_avg_example(together, CONFIG)

    if CONFIG["run_examples"]["hmm_trailing"]:
        if higher_frequency_data is None:
            raise FileNotFoundError(
                f"Missing required file for trailing SL example: {HIGHER_FREQUENCY_PATH}"
            )
        run_hmm_trailing_example(together, higher_frequency_data, CONFIG)

    if CONFIG["run_examples"]["hmm_avg"]:
        run_hmm_avg_example(together, CONFIG)

    if CONFIG["run_examples"]["kalman"]:
        run_kalman_example(together)

    if CONFIG["run_examples"]["metrics"]:
        # Example: metrics on the last computed strategy curve (if present)
        # Add your curve here if you want metrics printed by default.
        pass


if __name__ == "__main__":
    main()
