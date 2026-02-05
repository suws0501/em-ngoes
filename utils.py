from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
import shap
from scipy import stats
from hmmlearn.hmm import GMMHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from pykalman import KalmanFilter


_ZIGZAG_PATH = Path(__file__).resolve().parent / "ZigZag"
if _ZIGZAG_PATH.exists() and str(_ZIGZAG_PATH) not in sys.path:
    sys.path.insert(0, str(_ZIGZAG_PATH))

try:
    from zigzag import peak_valley_pivots
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "Failed to import `zigzag`. Ensure the ZigZag package is installed "
        "or the local `ZigZag/` directory exists."
    ) from exc


DEBUG = True
PLOT_SAVE = False
PLOT_DIR = "plots"
_PLOT_COUNTER = 0


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)


def set_plot_config(save_plots: bool, plots_dir: str = "plots") -> None:
    global PLOT_SAVE, PLOT_DIR
    PLOT_SAVE = bool(save_plots)
    PLOT_DIR = plots_dir


def show_or_save_plot(name: str) -> None:
    global _PLOT_COUNTER
    if not PLOT_SAVE:
        plt.show()
        return
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    _PLOT_COUNTER += 1
    safe_name = name.replace(" ", "_").replace("/", "_")
    out = Path(PLOT_DIR) / f"{_PLOT_COUNTER:03d}_{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def debug_block(title: str, **items) -> None:
    if not DEBUG:
        return
    line = "=" * 72
    print(f"\n{line}\n[DEBUG] {title}\n{line}")
    for key, value in items.items():
        if isinstance(value, (list, tuple, set)) and len(value) > 12:
            preview = list(value)[:12]
            print(f"- {key}: {preview} ... ({len(value)} total)")
        else:
            print(f"- {key}: {value}")


# === Feature Engineering ===

def getIndicators(price: pd.DataFrame, timestamp: bool):
    def trends(df: pd.DataFrame, indicators: pd.DataFrame, timestamp: bool = True):
        adx = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
        )
        indicators["adx"] = adx.adx()
        indicators["adx_neg"] = adx.adx_neg()
        indicators["adx_pos"] = adx.adx_pos()

        aroon = ta.trend.AroonIndicator(
            high=df["high"], low=df["low"], window=25, fillna=False
        )
        indicators["aroon_down"] = aroon.aroon_down()
        indicators["aroon_up"] = aroon.aroon_up()
        indicators["aroon_indicator"] = aroon.aroon_indicator()

        cci = ta.trend.CCIIndicator(
            high=df.high, low=df.low, close=df.close, window=20, constant=0.015, fillna=False
        )
        indicators["cci"] = cci.cci()

        dpo = ta.trend.DPOIndicator(close=df.close, window=20, fillna=False)
        indicators["dpo"] = dpo.dpo()

        ema = ta.trend.EMAIndicator(close=df["close"], window=14, fillna=False)
        indicators["ema_indicator"] = ema.ema_indicator()
        indicators["ema_diff_indicator"] = ema.ema_indicator() - df["close"]

        ichimoku = ta.trend.IchimokuIndicator(
            high=df["high"],
            low=df["low"],
            window1=9,
            window2=26,
            window3=52,
            visual=False,
            fillna=False,
        )
        indicators["ichimoku_a"] = ichimoku.ichimoku_a()
        indicators["ichimoku_b"] = ichimoku.ichimoku_b()
        indicators["ichimoku_base_line"] = ichimoku.ichimoku_base_line()
        indicators["ichimoku_conversion_line"] = ichimoku.ichimoku_conversion_line()

        kst = ta.trend.KSTIndicator(
            close=df["close"],
            roc1=10,
            roc2=15,
            roc3=20,
            roc4=30,
            window1=10,
            window2=10,
            window3=10,
            window4=15,
            nsig=9,
            fillna=False,
        )
        indicators["kst"] = kst.kst()
        indicators["kst_diff"] = kst.kst_diff()
        indicators["kst_sig"] = kst.kst_sig()

        macd = ta.trend.MACD(
            close=df.close, window_slow=26, window_fast=12, window_sign=9, fillna=False
        )
        indicators["macd"] = macd.macd()
        indicators["macd_diff"] = macd.macd_diff()
        indicators["macd_signal"] = macd.macd_signal()

        mass_index = ta.trend.MassIndex(
            high=df["high"], low=df["low"], window_fast=9, window_slow=25, fillna=False
        )
        indicators["mass_index"] = mass_index.mass_index()

        psar = ta.trend.PSARIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            step=0.02,
            max_step=0.2,
            fillna=False,
        )
        indicators["psar"] = df["close"] - psar.psar()
        indicators["psar_down"] = (df["close"] - psar.psar_down()).fillna(0)
        indicators["psar_down_indicator"] = psar.psar_down_indicator()
        indicators["psar_up"] = (psar.psar_up() - df["close"]).fillna(0)
        indicators["psar_up_indicator"] = psar.psar_up_indicator()

        sma = ta.trend.SMAIndicator(close=df["close"], window=30, fillna=False)
        indicators["sma_indicator"] = df["close"] - sma.sma_indicator()

        indicators["stc"] = ta.trend.STCIndicator(
            close=df.close, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3, fillna=False
        ).stc()

        indicators["trix"] = ta.trend.TRIXIndicator(
            close=df.close, window=15, fillna=False
        ).trix()

        vortex = ta.trend.VortexIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14, fillna=False
        )
        indicators["vortex_indicator_diff"] = vortex.vortex_indicator_diff()
        indicators["vortex_indicator_neg"] = vortex.vortex_indicator_neg()
        indicators["vortex_indicator_pos"] = vortex.vortex_indicator_pos()

        if timestamp:
            indicators["timestamp"] = df["timestamp"]
        return indicators

    def momentum(df: pd.DataFrame, indicators: pd.DataFrame, timestamp: bool = True):
        indicators["RSI"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()

        stoch = ta.momentum.StochasticOscillator(
            low=df["low"], high=df["high"], close=df["close"], window=14
        )
        indicators["stochatsic_oscillator"] = stoch.stoch()
        indicators["stochatsic_oscillator_signal"] = stoch.stoch_signal()

        indicators["awesome_oscillator"] = ta.momentum.AwesomeOscillatorIndicator(
            high=df["high"], low=df["low"], window1=5, window2=34, fillna=False
        ).awesome_oscillator()

        indicators["kama_indicator"] = (
            ta.momentum.KAMAIndicator(
                close=df["close"], window=10, pow1=2, pow2=30, fillna=False
            ).kama()
            - df["close"]
        )

        ppo = ta.momentum.PercentagePriceOscillator(
            close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
        )
        indicators["ppo"] = ppo.ppo()
        indicators["ppo_signal"] = ppo.ppo_signal()

        pvo = ta.momentum.PercentageVolumeOscillator(
            volume=df["volume"], window_slow=26, window_fast=12, window_sign=9, fillna=False
        )
        indicators["pvo"] = pvo.pvo()
        indicators["pvo_signal"] = pvo.pvo_signal()

        roc = ta.momentum.ROCIndicator(close=df["close"], window=12, fillna=False)
        indicators["roc"] = roc.roc()

        stochrsi = ta.momentum.StochRSIIndicator(
            close=df["close"], window=14, smooth1=3, smooth2=3, fillna=False
        )
        indicators["stochrsi"] = stochrsi.stochrsi()
        indicators["stochrsi_d"] = stochrsi.stochrsi_d()
        indicators["stochrsi_k"] = stochrsi.stochrsi_k()

        indicators["true_strength_index"] = ta.momentum.TSIIndicator(
            close=df["close"], window_slow=25, window_fast=13, fillna=False
        ).tsi()

        indicators["Ultimate_Indicator"] = ta.momentum.UltimateOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window1=7,
            window2=14,
            window3=28,
            weight1=4.0,
            weight2=2.0,
            weight3=1.0,
            fillna=False,
        ).ultimate_oscillator()

        indicators["Williams_R_Indicator"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"], lbp=14, fillna=False
        ).williams_r()

        if timestamp:
            indicators["timestamp"] = df["timestamp"]
        return indicators

    def volatility(df: pd.DataFrame, indicators: pd.DataFrame, timestamp: bool = True):
        indicators["atr"] = ta.volatility.average_true_range(
            high=df.high, low=df.low, close=df.close, window=14, fillna=False
        )

        bollinger = ta.volatility.BollingerBands(
            close=df["close"], window=20, window_dev=2, fillna=False
        )
        indicators["bollinger_upperband_price_diff"] = df["close"] - bollinger.bollinger_hband()
        indicators["bollinger_hband_indicator"] = bollinger.bollinger_hband_indicator()
        indicators["bollinger_lowerband_price_diff"] = bollinger.bollinger_lband() - df["close"]
        indicators["bollinger_lband_indicator"] = bollinger.bollinger_lband_indicator()
        indicators["bollinger_mavg_price_diff"] = bollinger.bollinger_mavg() - df["close"]
        indicators["bollinger_pband"] = bollinger.bollinger_pband()
        indicators["bollinger_wband"] = bollinger.bollinger_wband()

        donchian = ta.volatility.DonchianChannel(
            high=df["high"], low=df["low"], close=df["close"], window=20, offset=0, fillna=False
        )
        indicators["donchian_channel_hband"] = donchian.donchian_channel_hband() - df["close"]
        indicators["donchian_channel_lband"] = df["close"] - donchian.donchian_channel_lband()
        indicators["donchian_channel_mband"] = donchian.donchian_channel_mband() - df["close"]
        indicators["donchian_channel_pband"] = donchian.donchian_channel_pband()
        indicators["donchian_channel_wband"] = donchian.donchian_channel_wband()

        keltnerchannel = ta.volatility.KeltnerChannel(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=20,
            window_atr=10,
            fillna=False,
            original_version=True,
            multiplier=2,
        )
        indicators["keltner_channel_hband"] = keltnerchannel.keltner_channel_hband() - df["close"]
        indicators["keltner_channel_hband_indicator"] = keltnerchannel.keltner_channel_hband_indicator()
        indicators["keltner_channel_lband"] = df["close"] - keltnerchannel.keltner_channel_lband()
        indicators["keltner_channel_lband_indicator"] = keltnerchannel.keltner_channel_lband_indicator()
        indicators["keltner_channel_mband"] = keltnerchannel.keltner_channel_mband() - df["close"]
        indicators["keltner_channel_pband"] = keltnerchannel.keltner_channel_pband()
        indicators["keltner_channel_wband"] = keltnerchannel.keltner_channel_wband()

        indicators["ulcer_index"] = ta.volatility.ulcer_index(
            close=df.close, window=14, fillna=False
        )

        if timestamp:
            indicators["timestamp"] = df["timestamp"]
        return indicators

    def volume(df: pd.DataFrame, indicators: pd.DataFrame, timestamp: bool = True):
        indicators["AccDistIndexIndicator"] = ta.volume.AccDistIndexIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            fillna=False,
        ).acc_dist_index()

        chaikin_money_flow = ta.volume.ChaikinMoneyFlowIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=20,
            fillna=False,
        )
        indicators["chain_money_flow"] = chaikin_money_flow.chaikin_money_flow()

        eom = ta.volume.EaseOfMovementIndicator(
            high=df["high"], low=df["low"], volume=df["volume"], window=14, fillna=False
        )
        indicators["EoM"] = eom.ease_of_movement()
        indicators["sma_EoM"] = eom.sma_ease_of_movement()

        indicators["force_index"] = ta.volume.ForceIndexIndicator(
            close=df["close"], volume=df["volume"], window=13, fillna=False
        ).force_index()

        indicators["money_flow_index"] = ta.volume.MFIIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            window=14,
            fillna=False,
        ).money_flow_index()

        indicators["negative_volume_index"] = ta.volume.NegativeVolumeIndexIndicator(
            close=df["close"], volume=df["volume"], fillna=False
        ).negative_volume_index()

        indicators["on_balance_volume"] = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"], fillna=False
        ).on_balance_volume()

        indicators["volume_price_trend"] = ta.volume.VolumePriceTrendIndicator(
            close=df["close"], volume=df["volume"], fillna=False
        ).volume_price_trend()

        indicators["volume_weightd_average_price"] = (
            ta.volume.VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                window=14,
                fillna=False,
            ).volume_weighted_average_price()
            - df["close"]
        )

        if timestamp:
            indicators["timestamp"] = df["timestamp"]
        return indicators

    indicators = pd.DataFrame()
    indicators = trends(price, indicators, timestamp=timestamp)
    indicators = momentum(price, indicators, timestamp=timestamp)
    indicators = volume(price, indicators, timestamp=timestamp)
    indicators = volatility(price, indicators, timestamp=timestamp)
    return indicators


def tsfresh_rolling_features(
    df: pd.DataFrame,
    time_series_col: str = "Close",
    window_size: int = 2500,
    start_idx: int = 11000,
    preset: str = "comprehensive",
    exclude: Optional[list] = None,
    n_jobs: int = 0,
) -> pd.DataFrame:
    if time_series_col not in df.columns:
        raise KeyError(f"`{time_series_col}` not found in df.columns")

    preset = preset.lower()
    if preset == "comprehensive":
        fc = ComprehensiveFCParameters()
    elif preset == "efficient":
        fc = EfficientFCParameters()
    elif preset == "minimal":
        fc = MinimalFCParameters()
    else:
        raise ValueError("preset must be one of {'comprehensive','efficient','minimal'}")

    if exclude is None:
        exclude = ["linear_trend", "agg_linear_trend"]
    for k in exclude:
        fc.pop(k, None)

    s = df[[time_series_col]].sort_index()[time_series_col]

    if len(s) < window_size:
        raise ValueError(f"Series length {len(s)} < window_size {window_size}.")

    start = max(start_idx, window_size - 1)

    windows = []
    for i in range(start, len(s)):
        win = s.iloc[i - window_size + 1 : i + 1]
        if win.isna().any():
            continue
        tmp = pd.DataFrame(
            {
                "time": np.arange(window_size, dtype=int),
                "value": win.values,
                "id": [s.index[i]] * window_size,
            }
        )
        windows.append(tmp)

    if not windows:
        raise RuntimeError("No valid windows were created. Check NaNs and indices.")

    full_df = pd.concat(windows, ignore_index=True)

    feats = extract_features(
        full_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc,
        n_jobs=n_jobs,
    )

    try:
        feats.index = pd.to_datetime(feats.index)
    finally:
        feats = feats.sort_index()

    return feats


def getEvent(price: pd.DataFrame, threshold: float = 0.01):
    df = price.copy()
    df["Event"] = peak_valley_pivots(df["close"], threshold, -threshold)
    target = df["Event"].shift(-1)
    return target


def compute_ATR(df: pd.DataFrame, window: int = 14, method: str = "ema"):
    data = df.copy()
    data["prev_close"] = data["close"].shift(1)
    tr1 = data["high"] - data["low"]
    tr2 = (data["high"] - data["prev_close"]).abs()
    tr3 = (data["low"] - data["prev_close"]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if method == "sma":
        atr = tr.rolling(window=window).mean()
    elif method == "ema":
        atr = tr.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")
    return atr


def get_full_dataset(data: pd.DataFrame, threshold: float = 0.01):
    debug_block(
        "get_full_dataset:start",
        rows=len(data),
        cols=len(data.columns),
        has_close="close" in data.columns,
        has_low="low" in data.columns,
        has_high="high" in data.columns,
    )
    together = getIndicators(data.copy(), False)
    event = getEvent(data.copy(), threshold)

    together["Event"] = event.values[:]
    together["price"] = data["close"].values[:]
    together["low"] = data["low"].values[:]
    together["high"] = data["high"].values[:]
    together["ATR"] = compute_ATR(data.copy(), window=20, method="ema")

    returns = together["price"].pct_change()
    together[f"ret_skew_{20}"] = returns.rolling(20).skew()

    together.dropna(inplace=True)
    debug_block(
        "get_full_dataset:end",
        rows=len(together),
        cols=len(together.columns),
        has_event="Event" in together.columns,
        has_atr="ATR" in together.columns,
    )
    return together


# === Feature Pruning & PCA Utilities ===

def keep_numeric_nonconstant(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.select_dtypes(include=[np.number]).copy()
    nunique = Xn.nunique(dropna=False)
    Xn = Xn.loc[:, nunique > 1]
    return Xn


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    Xn = keep_numeric_nonconstant(X).dropna(axis=0)
    if Xn.shape[1] <= 1:
        return pd.DataFrame({"feature": Xn.columns, "vif": [1.0] * Xn.shape[1]})

    arr = Xn.to_numpy()
    vifs = []
    for i in range(arr.shape[1]):
        vifs.append(variance_inflation_factor(arr, i))
    out = pd.DataFrame({"feature": Xn.columns, "vif": vifs}).sort_values("vif", ascending=False)
    return out


def vif_prune_iterative(X: pd.DataFrame, vif_thresh: float = 10.0, max_iter: int = 100):
    Xc = keep_numeric_nonconstant(X).copy()
    removed = []

    for _ in range(max_iter):
        if Xc.shape[1] <= 1:
            break
        vif_table = compute_vif(Xc)
        if vif_table.empty:
            break

        top_feat = vif_table.iloc[0]["feature"]
        top_vif = float(vif_table.iloc[0]["vif"])

        if np.isnan(top_vif) or np.isinf(top_vif) or top_vif > vif_thresh:
            Xc = Xc.drop(columns=[top_feat])
            removed.append(top_feat)
        else:
            break

    final_vif = (
        compute_vif(Xc)
        if Xc.shape[1] >= 2
        else pd.DataFrame({"feature": Xc.columns, "vif": [1.0] * Xc.shape[1]})
    )
    return Xc, removed, final_vif


def mi_feature_target(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str = "regression",
    n_neighbors: int = 5,
    random_state: int = 42,
) -> pd.Series:
    Xn = keep_numeric_nonconstant(X)
    y_arr = np.asarray(y)

    mask = Xn.notna().all(axis=1) & ~pd.isna(y_arr)
    Xc = Xn.loc[mask]
    yc = y_arr[mask]

    if Xc.shape[0] == 0:
        return pd.Series(0.0, index=Xn.columns)

    if task == "classification":
        mi = mutual_info_classif(
            Xc, yc, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state
        )
    else:
        mi = mutual_info_regression(
            Xc, yc, discrete_features=False, n_neighbors=n_neighbors, random_state=random_state
        )

    return pd.Series(mi, index=Xc.columns).reindex(Xn.columns).fillna(0.0)


def pairwise_mi_symmetric(
    X: pd.DataFrame, n_neighbors: int = 5, random_state: int = 42
) -> pd.DataFrame:
    Xn = keep_numeric_nonconstant(X)
    Xc = Xn.dropna(axis=0)
    cols = list(Xc.columns)
    n = len(cols)

    if n == 0:
        return pd.DataFrame()
    if n == 1:
        return pd.DataFrame([[0.0]], index=cols, columns=cols)

    arr = Xc.to_numpy()
    M = np.zeros((n, n), dtype=float)

    def _mi_1d(x, y):
        return float(
            mutual_info_regression(
                x.reshape(-1, 1),
                y,
                discrete_features=False,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
        )

    for i in range(n):
        for j in range(i + 1, n):
            mi_ij = _mi_1d(arr[:, i], arr[:, j])
            mi_ji = _mi_1d(arr[:, j], arr[:, i])
            mij = 0.5 * (mi_ij + mi_ji)
            M[i, j] = mij
            M[j, i] = mij

    return pd.DataFrame(M, index=cols, columns=cols)


def corr_mi_prune(
    X: pd.DataFrame,
    corr_thresh: float = 0.90,
    mi_thresh: Optional[float] = None,
    y: Optional[np.ndarray] = None,
    task: str = "regression",
    n_neighbors: int = 5,
    random_state: int = 42,
):
    Xn = keep_numeric_nonconstant(X).copy()
    if Xn.shape[1] <= 1:
        return Xn, []

    Xc = Xn.dropna(axis=0)
    if Xc.shape[1] <= 1 or Xc.shape[0] < 5:
        return Xn, []

    corr = Xc.corr().abs()
    mi_mat = (
        pairwise_mi_symmetric(Xc, n_neighbors=n_neighbors, random_state=random_state)
        if mi_thresh is not None
        else None
    )
    var = Xc.var()

    rel = None
    if y is not None:
        rel = mi_feature_target(Xn, y, task=task, n_neighbors=n_neighbors, random_state=random_state)

    cols = list(Xc.columns)
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = float(corr.loc[a, b])
            m = float(mi_mat.loc[a, b]) if mi_mat is not None else -np.inf

            if (c >= corr_thresh) or (mi_mat is not None and m >= mi_thresh):
                pairs.append((a, b, c, m))

    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)

    keep = set(Xn.columns)
    removed = []

    def choose_drop(a, b):
        if rel is not None:
            return a if rel[a] < rel[b] else b
        return a if var.get(a, 0.0) < var.get(b, 0.0) else b

    for a, b, _, _ in pairs:
        if a not in keep or b not in keep:
            continue
        d = choose_drop(a, b)
        keep.remove(d)
        removed.append(d)

    Xout = Xn.loc[:, [c for c in Xn.columns if c in keep]]
    return Xout, removed


def fit_group_pca(
    X: pd.DataFrame, cols: list[str], standardize: bool = True, n_components: int = 1
):
    Xg = X[cols].copy()
    Xg = Xg.dropna(axis=0)
    if Xg.shape[0] < 5 or Xg.shape[1] < 2:
        return None, None, cols

    scaler = None
    Z = Xg.to_numpy()
    if standardize:
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(Z)
    return scaler, pca, cols


def transform_group_pca(
    X: pd.DataFrame,
    cols: list[str],
    scaler,
    pca,
    out_prefix: str = "group",
    n_components: int = 1,
):
    Xg = X[cols].copy()
    mask = Xg.notna().all(axis=1)

    Zout = np.full((X.shape[0], n_components), np.nan, dtype=float)
    if scaler is None or pca is None:
        return pd.DataFrame(index=X.index)

    Xcomplete = Xg.loc[mask]
    if len(Xcomplete) > 0:
        Z = Xcomplete.to_numpy()
        if scaler is not None:
            Z = scaler.transform(Z)
        Zout[mask.values, :] = pca.transform(Z)

    pc_cols = [f"{out_prefix}_pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(Zout, index=X.index, columns=pc_cols)


def run_pipeline_fit_transform(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_list_to_transform: list[pd.DataFrame],
    groups: dict[str, list[str]],
    vif_thresh: float = 10.0,
    corr_thresh: float = 0.90,
    mi_thresh: Optional[float] = 0.05,
    task: str = "regression",
    mi_neighbors: int = 5,
    random_state: int = 42,
    pca_standardize: bool = True,
    pca_n_components: int = 1,
):
    X_train_num = keep_numeric_nonconstant(X_train)

    logs = {
        "vif_removed": {},
        "corr_mi_removed": {},
        "kept_cols": {},
        "pca_explained_var": {},
    }

    fitted = {}

    for gname, cols in groups.items():
        cols_present = [c for c in cols if c in X_train_num.columns]
        if len(cols_present) == 0:
            continue

        Xg0 = X_train_num[cols_present]
        Xg1, removed_vif, _ = vif_prune_iterative(Xg0, vif_thresh=vif_thresh)
        logs["vif_removed"][gname] = removed_vif

        Xg2, removed_cm = corr_mi_prune(
            Xg1,
            corr_thresh=corr_thresh,
            mi_thresh=mi_thresh,
            y=y_train,
            task=task,
            n_neighbors=mi_neighbors,
            random_state=random_state,
        )
        logs["corr_mi_removed"][gname] = removed_cm

        kept = list(Xg2.columns)
        logs["kept_cols"][gname] = kept

        scaler, pca, used_cols = fit_group_pca(
            X_train_num,
            kept,
            standardize=pca_standardize,
            n_components=pca_n_components,
        )

        if pca is not None:
            logs["pca_explained_var"][gname] = float(pca.explained_variance_ratio_[0])
            fitted[gname] = {"cols": used_cols, "scaler": scaler, "pca": pca}
        else:
            fitted[gname] = {"cols": used_cols, "scaler": None, "pca": None}

    all_group_cols = {c for cols in groups.values() for c in cols}
    nongroup_cols = [c for c in X_train_num.columns if c not in all_group_cols]

    out_parts_train = []
    if len(nongroup_cols) > 0:
        out_parts_train.append(X_train_num[nongroup_cols])

    for gname, obj in fitted.items():
        cols_used = obj["cols"]
        if obj["pca"] is not None:
            pc = transform_group_pca(
                X_train_num,
                cols_used,
                obj["scaler"],
                obj["pca"],
                out_prefix=gname,
                n_components=pca_n_components,
            )
            out_parts_train.append(pc)
        else:
            out_parts_train.append(X_train_num[cols_used])

    X_train_out = pd.concat(out_parts_train, axis=1)

    X_out_list = []
    for X_any in X_list_to_transform:
        Xn = keep_numeric_nonconstant(X_any)

        parts = []
        if len(nongroup_cols) > 0:
            cols_exist = [c for c in nongroup_cols if c in Xn.columns]
            if len(cols_exist) > 0:
                parts.append(Xn[cols_exist])

        for gname, obj in fitted.items():
            cols_used = [c for c in obj["cols"] if c in Xn.columns]
            if len(cols_used) == 0:
                continue

            if obj["pca"] is not None and len(cols_used) == len(obj["cols"]):
                pc = transform_group_pca(
                    Xn,
                    cols_used,
                    obj["scaler"],
                    obj["pca"],
                    out_prefix=gname,
                    n_components=pca_n_components,
                )
                parts.append(pc)
            else:
                parts.append(Xn[cols_used])

        X_out_list.append(pd.concat(parts, axis=1) if len(parts) else pd.DataFrame(index=X_any.index))

    return X_train_out, X_out_list, logs


# === Train/Test Utilities & Models ===

def embargo_split(X: pd.DataFrame, test_proportion: float, embargo_period: int):
    size = int(len(X) * test_proportion)
    train_size = len(X) - size
    return X.iloc[:train_size], X.iloc[train_size + embargo_period :]


def plot_feature_hist(x, title: str):
    x = np.asarray(x)

    plt.figure(figsize=(8, 5))
    plt.hist(
        x,
        bins=60,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="black",
        label="Histogram",
    )

    mu, sigma = np.mean(x), np.std(x)
    xx = np.linspace(x.min(), x.max(), 400)
    plt.plot(xx, stats.norm.pdf(xx, mu, sigma), "r", linewidth=2, label=f"Normal fit (μ={mu:.4g}, σ={sigma:.4g})")

    kde = stats.gaussian_kde(x)
    plt.plot(xx, kde(xx), "g", linewidth=2, label="KDE fit")

    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    show_or_save_plot(f"feature_hist_{title}")


def RF_meta_label_model(
    base_reg_df: pd.DataFrame,
    meta_reg_df: pd.DataFrame,
    top_k_shap: int,
    feat_cols: list,
    shap_sample_size: int = 100,
    alpha: float = 0.5,
    reducing_n_estimators: int = 1000,
    base_n_estimators: int = 200,
    meta_n_estimators: int = 100,
    early_stop: Optional[dict] = None,
    target_precision: Optional[float] = None,
    min_precision_coverage: float = 0.05,
):
    debug_block(
        "RF_meta_label_model:start",
        base_rows=len(base_reg_df),
        meta_rows=len(meta_reg_df),
        feat_cols=len(feat_cols),
        top_k_shap=top_k_shap,
        alpha=alpha,
    )
    train_bin = base_reg_df[base_reg_df["Event"].isin([-1, 1])].copy()
    if len(train_bin) == 0:
        raise ValueError("No labeled samples (-1/1) in base_reg_df for training.")

    reducing_feat_model = RandomForestClassifier(n_estimators=reducing_n_estimators, random_state=42)
    reducing_feat_model.fit(train_bin[feat_cols], train_bin["Event"])

    sample_n = min(shap_sample_size, len(train_bin))
    sample_X = train_bin[feat_cols].sample(sample_n, random_state=42)

    explainer = shap.TreeExplainer(reducing_feat_model)
    shap_values = explainer.shap_values(sample_X)

    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=1)
    shap_ranking = pd.Series(mean_abs_shap, index=train_bin[feat_cols].columns).sort_values(ascending=False)
    top_k_features = shap_ranking.head(top_k_shap).index.tolist()

    base_model = RandomForestClassifier(
        n_estimators=0 if early_stop else base_n_estimators,
        random_state=4,
        warm_start=bool(early_stop),
    )

    def _fit_with_early_stop(X, y):
        cfg = early_stop or {}
        val_frac = float(cfg.get("val_frac", 0.2))
        patience = int(cfg.get("patience", 3))
        min_delta = float(cfg.get("min_delta", 1e-4))
        step = int(cfg.get("step", 50))
        max_estimators = int(cfg.get("max_estimators", base_n_estimators))

        if len(X) < 200 or val_frac <= 0 or val_frac >= 0.5:
            base_model.set_params(n_estimators=max_estimators)
            base_model.fit(X, y)
            return

        split_idx = int(len(X) * (1 - val_frac))
        X_tr, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_tr, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        if len(X_val) < 50 or len(np.unique(y_val)) < 2:
            base_model.set_params(n_estimators=max_estimators)
            base_model.fit(X, y)
            return

        best_auc = -np.inf
        bad = 0
        n_estimators = 0

        while n_estimators < max_estimators:
            n_estimators = min(n_estimators + step, max_estimators)
            base_model.set_params(n_estimators=n_estimators)
            base_model.fit(X_tr, y_tr)
            proba = base_model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score((y_val == 1).astype(int), proba)
            if auc > best_auc + min_delta:
                best_auc = auc
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if early_stop:
        _fit_with_early_stop(train_bin[top_k_features], train_bin["Event"])
    else:
        base_model.fit(train_bin[top_k_features], train_bin["Event"])

    base_preds_meta = base_model.predict(meta_reg_df[top_k_features])
    base_probs_meta = base_model.predict_proba(meta_reg_df[top_k_features]).max(axis=1)

    probs_base = base_model.predict_proba(base_reg_df[top_k_features])
    max_probs_base = probs_base.max(axis=1)

    def _threshold_for_precision(probs, y_true, precision_target, min_keep_frac):
        order = np.argsort(probs)[::-1]
        y = y_true[order]
        p = probs[order]
        correct = (y == 1).astype(int)
        cum_correct = np.cumsum(correct)
        cum_total = np.arange(1, len(p) + 1)
        precision = cum_correct / cum_total
        min_keep = max(1, int(len(p) * min_keep_frac))
        idx = np.where((precision >= precision_target) & (cum_total >= min_keep))[0]
        if len(idx) == 0:
            return None
        return p[idx[-1]]

    threshold = None
    if target_precision is not None:
        y_base = (base_reg_df["Event"].values == base_model.predict(base_reg_df[top_k_features])).astype(int)
        threshold = _threshold_for_precision(
            max_probs_base, y_base, float(target_precision), float(min_precision_coverage)
        )

    if threshold is None:
        conformal_score = 1 - max_probs_base
        threshold = np.quantile(conformal_score, 1 - alpha)

    adjusted_preds = np.where(base_probs_meta >= 1 - threshold, base_preds_meta, 0)

    meta_remain = meta_reg_df[top_k_features + ["Event"]].copy()
    meta_remain["signals"] = adjusted_preds
    meta_remain["meta_feature"] = np.where(meta_remain["signals"] == meta_remain["Event"], 1, 0)

    meta_model = RandomForestClassifier(n_estimators=meta_n_estimators, random_state=42)
    meta_model.fit(meta_remain[top_k_features], meta_remain["meta_feature"])

    debug_block(
        "RF_meta_label_model:end",
        train_bin=len(train_bin),
        top_k_features=top_k_features,
        threshold=float(threshold),
    )
    return meta_model, base_model, threshold, top_k_features, reducing_feat_model, sample_X


def run_hmm_rf_pipeline_from_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embargo_period: int = 30,
    n_components: int = 3,
    n_mix: int = 4,
    hmm_iter: int = 300,
    hmm_seed: int = 100,
    top_k_shap: int = 20,
    shap_sample_size: int = 100,
    alpha: float = 0.5,
    skew_col: str = "ret_skew_20",
    skew_threshold: float = 0.5,
    reducing_n_estimators: int = 1000,
    base_n_estimators: int = 200,
    meta_n_estimators: int = 100,
    early_stop: Optional[dict] = None,
    target_precision: Optional[float] = None,
    min_precision_coverage: float = 0.05,
):
    debug_block(
        "run_hmm_rf_pipeline_from_split:start",
        train_rows=len(train_df),
        test_rows=len(test_df),
        n_components=n_components,
        n_mix=n_mix,
        hmm_iter=hmm_iter,
        top_k_shap=top_k_shap,
        skew_col=skew_col,
    )
    def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "returns" not in df:
            df["returns"] = df["price"].pct_change()
        if "log_ret2" not in df:
            df["log_ret2"] = np.log(df["returns"] ** 2 + 1e-8)
        df.dropna(subset=["returns"], inplace=True)
        return df

    train_df = _ensure_features(train_df)
    test_df = _ensure_features(test_df)

    base_model_df, meta_model_df = embargo_split(
        train_df.copy(), test_proportion=0.3, embargo_period=embargo_period
    )
    debug_block(
        "run_hmm_rf_pipeline_from_split:split",
        base_rows=len(base_model_df),
        meta_rows=len(meta_model_df),
    )

    returns_train = base_model_df[["returns"]].dropna().values
    hmm_model = GMMHMM(
        n_components=n_components,
        n_mix=n_mix,
        covariance_type="diag",
        n_iter=hmm_iter,
        random_state=hmm_seed,
    ).fit(returns_train)

    def _infer_regimes_on(df_part: pd.DataFrame) -> pd.DataFrame:
        x = df_part[["returns"]].values
        states = hmm_model.predict(x)

        out = df_part.copy()
        out["regime"] = states
        return out

    base_model_df = _infer_regimes_on(base_model_df)
    meta_model_df = _infer_regimes_on(meta_model_df)
    test_df = _infer_regimes_on(test_df)

    regime_vol = base_model_df.groupby("regime")["returns"].std().sort_values()
    debug_block(
        "run_hmm_rf_pipeline_from_split:regimes",
        regime_vol=regime_vol.to_dict(),
    )

    def relabel_regimes(df: pd.DataFrame, regime_vol: pd.Series) -> pd.DataFrame:
        low_vol_regime = regime_vol.index[0]
        mid_vol_regime = regime_vol.index[1] if len(regime_vol) > 2 else None
        high_vol_regime = regime_vol.index[-1]

        regime_label_map = {
            low_vol_regime: "low_vol_regime",
            high_vol_regime: "high_vol_regime",
        }
        if mid_vol_regime is not None:
            regime_label_map[mid_vol_regime] = "mid_vol_regime"

        out = df.copy()
        out["regime_relabel"] = out["regime"].map(regime_label_map)
        return out

    base_model_df = relabel_regimes(base_model_df, regime_vol)
    meta_model_df = relabel_regimes(meta_model_df, regime_vol)
    test_df = relabel_regimes(test_df, regime_vol)

    drop_cols = {
        "Event",
        "price",
        "regime",
        "returns",
        "low",
        "high",
        "ATR",
        "regime_relabel",
        "log_ret2",
    }
    feat_cols = [c for c in train_df.columns if (c not in drop_cols and not c.startswith("regime_p"))]

    meta_model, base_model, threshold, top_k_features, reducing_feat_model, sample_X = RF_meta_label_model(
        base_reg_df=base_model_df,
        meta_reg_df=meta_model_df,
        top_k_shap=top_k_shap,
        feat_cols=feat_cols,
        shap_sample_size=shap_sample_size,
        alpha=alpha,
        reducing_n_estimators=reducing_n_estimators,
        base_n_estimators=base_n_estimators,
        meta_n_estimators=meta_n_estimators,
        early_stop=early_stop,
        target_precision=target_precision,
        min_precision_coverage=min_precision_coverage,
    )

    use_cols = [c for c in top_k_features if c in test_df.columns]
    testset = test_df.copy()

    testset["meta_label_probs"] = meta_model.predict_proba(testset[use_cols])[:, 1]

    base_preds = base_model.predict(testset[use_cols])
    base_probs = base_model.predict_proba(testset[use_cols]).max(axis=1)

    testset["signals_raw"] = np.where(base_probs >= 1 - threshold, base_preds, 0)

    if skew_col not in testset.columns:
        raise KeyError(
            f"Column '{skew_col}' not found in testset. "
            "Make sure you computed rolling skewness in your feature construction step."
        )

    cond_block = (testset["regime_relabel"] == "high_vol_regime") & (
        abs(testset[skew_col]) > skew_threshold
    )

    testset["signals"] = np.where(cond_block, 0, testset["signals_raw"])

    testset.index = pd.to_datetime(testset.index)
    testset = testset.sort_index()

    labels = np.array([-1, 0, 1])
    cm = confusion_matrix(testset["Event"], testset["signals"], labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (HOLD=0, HMM + Skewness Filter)")
    show_or_save_plot("confusion_matrix_hmm_skew_filter")

    models = {
        "base": base_model,
        "meta": meta_model,
        "threshold": threshold,
        "features": top_k_features,
        "hmm_model": hmm_model,
    }

    debug_block(
        "run_hmm_rf_pipeline_from_split:end",
        testset_rows=len(testset),
        signals_counts=testset["signals"].value_counts().to_dict(),
    )
    return {
        "base_model_df": base_model_df,
        "meta_model_df": meta_model_df,
        "test_df": test_df,
        "models": models,
        "testset": testset,
    }


def run_rf_pipeline_from_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embargo_period: int = 30,
    top_k_shap: int = 20,
    shap_sample_size: int = 200,
    alpha: float = 0.5,
    reducing_n_estimators: int = 1000,
    base_n_estimators: int = 200,
    meta_n_estimators: int = 100,
    early_stop: Optional[dict] = None,
    target_precision: Optional[float] = None,
    min_precision_coverage: float = 0.05,
):
    debug_block(
        "run_rf_pipeline_from_split:start",
        train_rows=len(train_df),
        test_rows=len(test_df),
        top_k_shap=top_k_shap,
        shap_sample_size=shap_sample_size,
        alpha=alpha,
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    base_model_df, meta_model_df = embargo_split(
        train_df.copy(), test_proportion=0.3, embargo_period=embargo_period
    )

    drop_cols = {"Event", "price", "regime", "returns", "low", "high", "ATR", "second_model_label"}
    feat_cols = [c for c in train_df.columns if (c not in drop_cols)]

    meta_m, base_m, thr, top_feats, reducing_feature_model, sample_X = RF_meta_label_model(
        base_model_df,
        meta_model_df,
        top_k_shap=top_k_shap,
        feat_cols=feat_cols,
        shap_sample_size=shap_sample_size,
        alpha=alpha,
        reducing_n_estimators=reducing_n_estimators,
        base_n_estimators=base_n_estimators,
        meta_n_estimators=meta_n_estimators,
        early_stop=early_stop,
        target_precision=target_precision,
        min_precision_coverage=min_precision_coverage,
    )

    test_df["meta_label_probs"] = meta_m.predict_proba(test_df[top_feats])[:, 1]

    base_preds = base_m.predict(test_df[top_feats])
    base_probs = base_m.predict_proba(test_df[top_feats]).max(axis=1)
    test_df["signals"] = np.where(base_probs >= 1 - thr, base_preds, 0)

    accuracy = accuracy_score(test_df["Event"], test_df["signals"])
    print(f"Accuracy: {accuracy}")

    labels = np.array([-1, 0, 1])
    cm = confusion_matrix(test_df["Event"], test_df["signals"], labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (with HOLD=0)")
    show_or_save_plot("confusion_matrix_rf")

    debug_block(
        "run_rf_pipeline_from_split:end",
        top_feats=top_feats,
        signals_counts=test_df["signals"].value_counts().to_dict(),
    )
    return test_df, reducing_feature_model, sample_X


def strategy_backtesting_trailing_sl(
    predictions,
    prices,
    higher_frequency_data,
    low,
    high,
    meta_probs,
    atr,
    sl_k: float = 2.0,
    tp_k: float = 5.0,
    prob_threshold: float = 0.3,
):
    debug_block(
        "strategy_backtesting_trailing_sl:start",
        bars=len(prices),
        sl_k=sl_k,
        tp_k=tp_k,
        prob_threshold=prob_threshold,
    )
    predictions = np.asarray(predictions)
    meta_probs = np.asarray(meta_probs)
    low = np.asarray(low)
    high = np.asarray(high)
    atr = np.asarray(atr)

    l_tp_prices = prices.values + tp_k * atr
    l_sl_prices = prices.values - sl_k * atr
    s_tp_prices = prices.values - tp_k * atr
    s_sl_prices = prices.values + sl_k * atr

    position = 0.0
    entry_idx = None
    entry_px = None
    entry_ts = None
    curr_sl = None
    fixed_tp = None

    equity = np.ones(len(prices), dtype=float)
    available_funds = 1.0
    trade_log = []

    def exit_long(l_sl_price, high_t1, low_t1, bar_start, bar_end, window):
        nonlocal curr_sl, fixed_tp
        candidate_sl = l_sl_price
        curr_sl = max(curr_sl, candidate_sl)
        sl, tp = curr_sl, fixed_tp
        hit_sl = low_t1 <= sl
        hit_tp = high_t1 >= tp
        if hit_sl and hit_tp:
            result, which, px, _ = resolve_with_lower_tf(bar_start, bar_end, "long", sl, tp, window)
            return result, px, which
        if hit_sl:
            return True, sl, "SL"
        if hit_tp:
            return True, tp, "TP"
        return False, None, None

    def exit_short(s_sl_price, high_t1, low_t1, bar_start, bar_end, window):
        nonlocal curr_sl, fixed_tp
        candidate_sl = s_sl_price
        curr_sl = min(curr_sl, candidate_sl)
        sl, tp = curr_sl, fixed_tp
        hit_sl = high_t1 >= sl
        hit_tp = low_t1 <= tp
        if hit_sl and hit_tp:
            result, which, px, _ = resolve_with_lower_tf(bar_start, bar_end, "short", sl, tp, window)
            return result, px, which
        if hit_sl:
            return True, sl, "SL"
        if hit_tp:
            return True, tp, "TP"
        return False, None, None

    def _resolve_micro_order(open_, high_, low_, close_, side, sl, tp, model="auto"):
        if side == "long":
            if open_ <= sl:
                return "SL", sl
            if open_ >= tp:
                return "TP", tp
        else:
            if open_ >= sl:
                return "SL", sl
            if open_ <= tp:
                return "TP", tp
        if model == "HL":
            return ("TP", tp) if side == "long" else ("SL", sl)
        if model == "LH":
            return ("SL", sl) if side == "long" else ("TP", tp)
        bullish = close_ >= open_
        if side == "long":
            return ("SL", sl) if bullish else ("TP", tp)
        return ("TP", tp) if bullish else ("SL", sl)

    def resolve_with_lower_tf(hourly_bar_start, hourly_bar_end, side, sl, tp, window, micro_model="auto"):
        for ts, row in window.iterrows():
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            if side == "long":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp
            if hit_tp and not hit_sl:
                return True, "TP", tp, ts
            if hit_sl and not hit_tp:
                return True, "SL", sl, ts
            if not hit_tp and not hit_sl:
                continue
            which, px = _resolve_micro_order(o, h, l, c, side=side, sl=sl, tp=tp, model=micro_model)
            return True, which, px, ts
        return False, None, None, None

    for t in range(1, len(predictions)):
        signal = predictions[t - 1]
        prob = float(meta_probs[t - 1])
        low_t1 = float(low[t])
        high_t1 = float(high[t])
        bar_start = prices.index[t]
        bar_end = bar_start + pd.Timedelta(hours=1)
        window = higher_frequency_data.loc[
            (higher_frequency_data.index >= bar_start) & (higher_frequency_data.index < bar_end)
        ]

        if entry_px is not None:
            l_sl_price = l_sl_prices[t - 1]
            s_sl_price = s_sl_prices[t - 1]
            if position > 0:
                exit_flag, exit_price, exit_reason = exit_long(
                    l_sl_price, high_t1, low_t1, bar_start, bar_end, window
                )
                if exit_flag:
                    ret = (exit_price - entry_px) / entry_px
                    available_funds *= 1 + ret * position
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": f"close long {exit_reason}",
                            "trade_price": exit_price,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None
            elif position < 0:
                exit_flag, exit_price, exit_reason = exit_short(
                    s_sl_price, high_t1, low_t1, bar_start, bar_end, window
                )
                if exit_flag:
                    ret = (entry_px - exit_price) / entry_px
                    available_funds *= 1 + ret * abs(position)
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": f"close short {exit_reason}",
                            "trade_price": exit_price,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

        if prob > prob_threshold:
            if signal == 1:
                if position > 0:
                    pnl = (prices.iloc[t] - entry_px) / entry_px * position
                    available_funds *= 1 + pnl
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close long (flip)",
                            "trade_price": float(prices.iloc[t]),
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

                if entry_px is None:
                    entry_idx = t
                    entry_px = float(prices.iloc[t])
                    entry_ts = prices.index[t]
                    fixed_tp = float(s_tp_prices[entry_idx])
                    curr_sl = float(s_sl_prices[entry_idx])
                else:
                    entry_px = (entry_px * abs(position) + float(prices.iloc[t]) * prob) / (
                        abs(position) + prob
                    )

                position -= prob
                trade_log.append(
                    {
                        "time": prices.index[t],
                        "trade": "short",
                        "trade_price": float(prices.iloc[t]),
                        "average_price": entry_px,
                        "trade_position": abs(prob),
                        "current_total_position": position,
                    }
                )

            elif signal == -1:
                if position < 0:
                    pnl = (entry_px - prices.iloc[t]) / entry_px * abs(position)
                    available_funds *= 1 + pnl
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close short (flip)",
                            "trade_price": float(prices.iloc[t]),
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

                if entry_px is None:
                    entry_idx = t
                    entry_px = float(prices.iloc[t])
                    entry_ts = prices.index[t]
                    fixed_tp = float(l_tp_prices[entry_idx])
                    curr_sl = float(l_sl_prices[entry_idx])
                else:
                    entry_px = (entry_px * abs(position) + float(prices.iloc[t]) * prob) / (
                        abs(position) + prob
                    )

                position += prob
                trade_log.append(
                    {
                        "time": prices.index[t],
                        "trade": "long",
                        "trade_price": float(prices.iloc[t]),
                        "average_price": entry_px,
                        "trade_position": abs(prob),
                        "current_total_position": position,
                    }
                )

        if entry_px is not None:
            if position > 0:
                mtm = (float(prices.iloc[t]) - entry_px) / entry_px * position
            else:
                mtm = (entry_px - float(prices.iloc[t])) / entry_px * abs(position)
            equity[t] = available_funds * (1 + mtm)
        else:
            equity[t] = available_funds

    returns_series = pd.Series(equity, index=prices.index)
    trade_df = pd.DataFrame(trade_log)
    debug_block(
        "strategy_backtesting_trailing_sl:end",
        trades=len(trade_df),
        final_equity=float(returns_series.iloc[-1]) if len(returns_series) else None,
    )
    return returns_series, trade_df


def strategy_backtesting_average_change_rate(
    predictions,
    prices,
    meta_probs,
    train_set_price_event,
    prob_threshold: float = 0.3,
):
    debug_block(
        "strategy_backtesting_average_change_rate:start",
        bars=len(prices),
        prob_threshold=prob_threshold,
    )
    predictions = np.asarray(predictions)
    meta_probs = np.asarray(meta_probs)

    def compute_avg_pivot_change(df):
        evt_col = "event" if "event" in df.columns else ("Event" if "Event" in df.columns else None)
        if evt_col is None:
            raise KeyError("train_set_price_event must contain 'event' or 'Event'.")
        px_col = "price" if "price" in df.columns else ("Price" if "Price" in df.columns else None)
        if px_col is None:
            raise KeyError("train_set_price_event must contain 'price' or 'Price'.")

        df = df.copy()
        df["event_shifted"] = df[evt_col].shift(1)

        change_rates = []
        last_pivot_price = None
        last_pivot_type = None
        for _, row in df.iterrows():
            event, price = row["event_shifted"], row[px_col]
            if pd.notna(event) and event != 0:
                if last_pivot_price is not None and last_pivot_type != event:
                    rate = (price - last_pivot_price) / last_pivot_price
                    change_rates.append(abs(rate))
                last_pivot_price, last_pivot_type = price, event

        avg_rate = float(np.mean(change_rates)) if change_rates else 0.0
        print(f"Average peak-valley absolute return: {avg_rate:.4%}")
        return avg_rate

    avg_change = compute_avg_pivot_change(train_set_price_event)
    avg_change = max(avg_change, 1e-4)

    position = 0.0
    entry_px = None
    equity = np.ones(len(prices), dtype=float)
    available_funds = 1.0
    trade_log = []

    for t in range(1, len(predictions)):
        signal = float(predictions[t - 1])
        prob = float(meta_probs[t - 1])
        cur_px = float(prices.iloc[t])

        if entry_px is not None:
            ret = (cur_px - entry_px) / entry_px

            if position > 0:
                if ret >= avg_change:
                    available_funds *= 1 + ret * position
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close long TP",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None
                elif ret <= -avg_change:
                    available_funds *= 1 + ret * position
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close long SL",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None

            elif position < 0:
                if ret <= -avg_change:
                    available_funds *= 1 + ret * position
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close short TP",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None
                elif ret >= avg_change:
                    available_funds *= 1 + ret * position
                    equity[t] = available_funds
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close short SL",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None

        if prob > prob_threshold:
            if signal == 1:
                if position > 0 and entry_px is not None:
                    ret = (cur_px - entry_px) / entry_px
                    available_funds *= 1 + ret * position
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close long (flip)",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None

                if entry_px is None:
                    entry_px = cur_px
                else:
                    entry_px = (entry_px * abs(position) + cur_px * prob) / (abs(position) + prob)
                position -= prob
                trade_log.append(
                    {
                        "time": prices.index[t],
                        "trade": "short",
                        "trade_price": cur_px,
                        "average_price": entry_px,
                        "trade_position": abs(prob),
                        "current_total_position": position,
                    }
                )

            elif signal == -1:
                if position < 0 and entry_px is not None:
                    ret = (cur_px - entry_px) / entry_px
                    available_funds *= 1 + ret * position
                    trade_log.append(
                        {
                            "time": prices.index[t],
                            "trade": "close short (flip)",
                            "trade_price": cur_px,
                            "average_price": entry_px,
                            "trade_position": abs(position),
                            "current_total_position": 0,
                        }
                    )
                    position, entry_px = 0.0, None

                if entry_px is None:
                    entry_px = cur_px
                else:
                    entry_px = (entry_px * abs(position) + cur_px * prob) / (abs(position) + prob)
                position += prob
                trade_log.append(
                    {
                        "time": prices.index[t],
                        "trade": "long",
                        "trade_price": cur_px,
                        "average_price": entry_px,
                        "trade_position": abs(prob),
                        "current_total_position": position,
                    }
                )

        if entry_px is not None:
            if position > 0:
                mtm = (float(prices.iloc[t]) - entry_px) / entry_px * position
            else:
                mtm = (entry_px - float(prices.iloc[t])) / entry_px * abs(position)
            equity[t] = available_funds * (1 + mtm)
        else:
            equity[t] = available_funds

    returns_series = pd.Series(equity, index=prices.index)
    trade_df = pd.DataFrame(trade_log)
    debug_block(
        "strategy_backtesting_average_change_rate:end",
        trades=len(trade_df),
        final_equity=float(returns_series.iloc[-1]) if len(returns_series) else None,
    )
    return returns_series, trade_df


def equity_curve_comparison(equity_curve_1, equity_curve_2, name_1, name_2):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_1, label=name_1, linewidth=2)
    plt.plot(equity_curve_2, label=name_2, linewidth=2)
    plt.title("Equity Curve Comparison")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    show_or_save_plot(f"equity_curve_{name_1}_vs_{name_2}")


def performance_metrics(equity_curve: pd.Series, periods_per_year: int = 252 * 6.5):
    debug_block(
        "performance_metrics:start",
        points=len(equity_curve),
        periods_per_year=periods_per_year,
    )
    returns = equity_curve.pct_change().dropna()

    mean_ret = returns.mean()
    vol = returns.std(ddof=1)
    sharpe = (mean_ret / vol * np.sqrt(periods_per_year)) if vol != 0 else np.nan

    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=1)
    sortino = (mean_ret / downside_vol * np.sqrt(periods_per_year)) if downside_vol != 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    total_periods = len(equity_curve) - 1
    if total_periods > 0:
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = total_periods / periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    else:
        total_return, cagr = np.nan, np.nan

    if max_dd is not None and max_dd != 0 and not np.isnan(max_dd) and not np.isnan(cagr):
        calmar = cagr / abs(max_dd)
    else:
        calmar = np.nan

    data = {
        "Total Return": [total_return],
        "CAGR": [cagr],
        "Sharpe": [sharpe],
        "Sortino": [sortino],
        "Max Drawdown": [max_dd],
        "Calmar": [calmar],
    }
    df = pd.DataFrame(data).T
    df.columns = ["Value"]
    df["Value_rounded"] = df["Value"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    debug_block(
        "performance_metrics:end",
        metrics=df["Value_rounded"].to_dict(),
    )
    return df


def kalman_filter_price(
    price,
    Q: float = 1e-5,
    R: float = 1e-2,
    x0=None,
    P0: float = 1.0,
    em: bool = False,
    em_iters: int = 20,
):
    if isinstance(price, (list, tuple, np.ndarray)):
        price = pd.Series(price)
    elif not isinstance(price, pd.Series):
        raise TypeError("price must be a 1D array-like object or pandas Series")

    y = price.astype(float).values
    y_obs = y.reshape(-1, 1)

    if x0 is None:
        x0 = float(y[0])

    kf = KalmanFilter(
        transition_matrices=np.array([[1.0]]),
        observation_matrices=np.array([[1.0]]),
        initial_state_mean=np.array([x0]),
        initial_state_covariance=np.array([[P0]]),
        transition_covariance=np.array([[Q]]),
        observation_covariance=np.array([[R]]),
    )

    if em:
        kf = kf.em(y_obs, n_iter=em_iters)

    state_means, state_covariances = kf.filter(y_obs)

    xhat = pd.Series(state_means[:, 0], index=price.index, name="kalman_state_mean")
    phat = pd.Series(state_covariances[:, 0, 0], index=price.index, name="kalman_state_variance")

    return xhat, phat, kf
