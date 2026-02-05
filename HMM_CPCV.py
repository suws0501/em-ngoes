# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 22:15:45 2025

@author: zhang
"""
import sys
# sys.path.append("C:/Users/zhang/Desktop/New folder/ZigZag/")  

from zigzag import *
# Commented out IPython magic to ensure Python compatibility.
#!git clone https://github.com/ambroseikpele/ZigZag
# %cd /content/ZigZag
#!pip install .
# %cd ..
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from hmmlearn import hmm
import shap
from scipy import stats
from hmmlearn.hmm import GMMHMM
from ib_insync import IB, Stock, util
import time
from dateutil.relativedelta import relativedelta
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import roc_auc_score
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from statsmodels.tsa.api import ARIMA, VAR, VECM, AutoReg
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from datetime import datetime as dt
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


#%%%
def getIndicators(price, timestamp):
    
    def trends(df, indicators, timestamp=True):

        ### TRENDS

        # ADX
        adx= ta.trend.ADXIndicator(high=df['high'], low= df['low'], close=df['close'], window= 14, fillna = False)
        indicators['adx']=adx.adx()
        indicators['adx_neg']=adx.adx_neg()
        indicators['adx_pos']=adx.adx_pos()

        # AROON
        aroon= ta.trend.AroonIndicator(high=df['high'], low=df['low'], window= 25, fillna = False)
        indicators['aroon_down']=aroon.aroon_down()
        indicators['aroon_up']=aroon.aroon_up()
        indicators['aroon_indicator']=aroon.aroon_indicator()

        # CCI
        cci= ta.trend.CCIIndicator(high=df.high, low=df.low, close=df.close, window=20, constant=0.015, fillna=False)
        indicators['cci']= cci.cci()


        # DPO
        dpo= ta.trend.DPOIndicator(close=df.close, window=20, fillna=False)
        indicators['dpo']= dpo.dpo()

        # EMA
        ema= ta.trend.EMAIndicator(close=df['close'], window=14, fillna = False)
        indicators['ema_indicator']= ema.ema_indicator()
        indicators['ema_diff_indicator']= ema.ema_indicator()-df['close']


        # IchimokuIndicator

        ichimoku= ta.trend.IchimokuIndicator(high=df['high'], low= df['low'], window1 = 9, window2= 26, window3 = 52, visual= False, fillna= False)
        indicators['ichimoku_a']= ichimoku.ichimoku_a()
        indicators['ichimoku_b']= ichimoku.ichimoku_b()
        indicators['ichimoku_base_line']= ichimoku.ichimoku_base_line()
        indicators['ichimoku_conversion_line']= ichimoku.ichimoku_conversion_line()


        # KST
        kst= ta.trend.KSTIndicator(close=df['close'], roc1 = 10, roc2 = 15, roc3= 20, roc4 = 30, window1= 10, window2= 10, window3= 10, window4= 15, nsig= 9, fillna= False)
        indicators['kst']=kst.kst()
        indicators['kst_diff']=kst.kst_diff()
        indicators['kst_sig']=kst.kst_sig()

        # MACD
        macd= ta.trend.MACD(close=df.close, window_slow= 26, window_fast = 12, window_sign= 9, fillna= False)
        indicators['macd']=macd.macd()
        indicators['macd_diff']=macd.macd_diff()
        indicators['macd_signal']=macd.macd_signal()

        # MassIndex
        mass_index=ta.trend.MassIndex(high= df['high'], low= df['low'], window_fast = 9, window_slow= 25, fillna = False)
        indicators['mass_index']=mass_index.mass_index()

        # PSAR
        psar= ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close'], step= 0.02, max_step = 0.2, fillna = False)
        indicators['psar']= df['close']-psar.psar()
        indicators['psar_down']= (df['close']-psar.psar_down()).fillna(0)
        indicators['psar_down_indicator']= psar.psar_down_indicator()
        indicators['psar_up']= (psar.psar_up()- df['close']).fillna(0)
        indicators['psar_up_indicator']= psar.psar_up_indicator()

        # SMA
        sma= ta.trend.SMAIndicator(close=df['close'], window=30, fillna = False)
        indicators['sma_indicator']= df['close']-sma.sma_indicator()

        # STC
        indicators['stc']=ta.trend.STCIndicator(close=df.close, window_slow= 50, window_fast=23, cycle= 10, smooth1= 3, smooth2 = 3, fillna = False).stc()

        # TRIX
        indicators['trix']=ta.trend.TRIXIndicator(close=df.close, window=15, fillna = False).trix()

        # VortexIndicator
        vortex= ta.trend.VortexIndicator(high=df['high'], low=df['low'], close=df['close'], window= 14, fillna= False)
        indicators['vortex_indicator_diff']= vortex.vortex_indicator_diff()
        indicators['vortex_indicator_neg']= vortex.vortex_indicator_neg()
        indicators['vortex_indicator_pos']= vortex.vortex_indicator_pos()
        if timestamp==True:
            indicators['timestamp']=df['timestamp']
        return indicators

    def momentum(df, indicators, timestamp=True):
        ### Momentum

        ### RSI
        indicators['RSI']= ta.momentum.RSIIndicator(close=df['close']).rsi()

        ### Stochatsic Oscillator
        stoch= ta.momentum.StochasticOscillator(low=df['low'], high=df['high'], close=df['close'], window=14)
        indicators['stochatsic_oscillator']= stoch.stoch()
        indicators['stochatsic_oscillator_signal']= stoch.stoch_signal()

        ### Awesome Oscillator
        indicators['awesome_oscillator']=ta.momentum.AwesomeOscillatorIndicator(high=df['high'], low=df['low'], window1= 5, window2= 34, fillna= False).awesome_oscillator()

        ### Kama Indicator
        indicators['kama_indicator']= ta.momentum.KAMAIndicator(close=df['close'], window= 10, pow1= 2, pow2= 30, fillna = False).kama()-df['close']

        ### Percentage Price Oscillator
        ppo= ta.momentum.PercentagePriceOscillator(close= df['close'], window_slow= 26, window_fast= 12, window_sign= 9, fillna = False)
        indicators['ppo']= ppo.ppo()
        indicators['ppo_signal']=ppo.ppo_signal()

        ### Percentage Volume Oscillator
        pvo= ta.momentum.PercentageVolumeOscillator(volume= df['volume'], window_slow= 26, window_fast= 12, window_sign= 9, fillna= False)
        indicators['pvo']= pvo.pvo()
        indicators['pvo_signal']= pvo.pvo_signal()

        ### Rate Of Change
        roc= ta.momentum.ROCIndicator(close= df['close'], window= 12, fillna= False)
        indicators['roc']= roc.roc()

        ### Stochastic RSI
        stochrsi=ta.momentum.StochRSIIndicator(close= df['close'], window = 14, smooth1 = 3, smooth2 = 3, fillna= False)
        indicators['stochrsi']=stochrsi.stochrsi()
        indicators['stochrsi_d']= stochrsi.stochrsi_d()
        indicators['stochrsi_k']= stochrsi.stochrsi_k()

        ### True Strength Index
        indicators['true_strength_index']=ta.momentum.TSIIndicator(close=df['close'], window_slow= 25, window_fast = 13, fillna = False).tsi()

        ### Ultimate Oscillator
        indicators['Ultimate_Indicator']=ta.momentum.UltimateOscillator(high= df['high'], low= df['low'], close=df['close'], window1= 7, window2 = 14, window3= 28, weight1 = 4.0, weight2 = 2.0, weight3 = 1.0, fillna = False).ultimate_oscillator()

        ### Williams_R_Indicator
        indicators['Williams_R_Indicator']=ta.momentum.WilliamsRIndicator(high= df['high'], low=df['low'], close=df['close'], lbp = 14, fillna = False).williams_r()
        if timestamp==True:
            indicators['timestamp']=df['timestamp']
        return indicators

    def volatility(df, indicators, timestamp=True):
        ### Volatility

        # ATR
        indicators['atr']=ta.volatility.average_true_range(high=df.high, low=df.low, close=df.close, window=14, fillna=False)

        # Bollinger Band
        bollinger=ta.volatility.BollingerBands(close=df['close'], window= 20, window_dev= 2, fillna = False)

        indicators['bollinger_upperband_price_diff']= df['close']-bollinger.bollinger_hband() # if positive, price is above upper band
        indicators['bollinger_hband_indicator']= bollinger.bollinger_hband_indicator()
        indicators['bollinger_lowerband_price_diff']= bollinger.bollinger_lband()- df['close']
        indicators['bollinger_lband_indicator']= bollinger.bollinger_lband_indicator()
        indicators['bollinger_mavg_price_diff']= bollinger.bollinger_mavg()- df['close']
        indicators['bollinger_pband']= bollinger.bollinger_pband()
        indicators['bollinger_wband']= bollinger.bollinger_wband()

        # DonchianChannel
        donchian= ta.volatility.DonchianChannel(high=df['high'], low=df['low'], close=df['close'], window = 20, offset = 0, fillna = False)
        indicators['donchian_channel_hband']= donchian.donchian_channel_hband()-df['close']
        indicators['donchian_channel_lband']= df['close']-donchian.donchian_channel_lband()
        indicators['donchian_channel_mband']= donchian.donchian_channel_mband()-df['close']
        indicators['donchian_channel_pband']= donchian.donchian_channel_pband()
        indicators['donchian_channel_wband']= donchian.donchian_channel_wband()

        # keltner channe
        keltnerchannel= ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window = 20, window_atr= 10, fillna = False, original_version = True, multiplier = 2)
        indicators['keltner_channel_hband']= keltnerchannel.keltner_channel_hband()-df['close']
        indicators['keltner_channel_hband_indicator']= keltnerchannel.keltner_channel_hband_indicator()
        indicators['keltner_channel_lband']= df['close']-keltnerchannel.keltner_channel_lband()
        indicators['keltner_channel_lband_indicator']= keltnerchannel.keltner_channel_lband_indicator()
        indicators['keltner_channel_mband']= keltnerchannel.keltner_channel_mband()-df['close']
        indicators['keltner_channel_pband']= keltnerchannel.keltner_channel_pband()
        indicators['keltner_channel_wband']= keltnerchannel.keltner_channel_wband()

        # Ulcer Index
        indicators['ulcer_index']=ta.volatility.ulcer_index(close=df.close, window=14, fillna=False)
        if timestamp==True:
            indicators['timestamp']=df['timestamp']
        return indicators

    def volume(df, indicators, timestamp=True):
        ### Volume

        ### AccDistIndexIndicator
        indicators['AccDistIndexIndicator']=ta.volume.AccDistIndexIndicator(high=df['high'], low= df['low'], close= df['close'], volume= df['volume'], fillna = False).acc_dist_index()

        ### Chaikin Money Flow Indicator
        chaikin_money_flow = ta.volume.ChaikinMoneyFlowIndicator(high=df['high'],low=df['low'],close=df['close'],volume=df['volume'],window=20,fillna=False)
        indicators['chain_money_flow']=chaikin_money_flow.chaikin_money_flow()

        ### Ease of Movement
        eom=ta.volume.EaseOfMovementIndicator(high=df['high'],low=df['low'],volume=df['volume'], window= 14, fillna = False)
        indicators['EoM']=eom.ease_of_movement()
        indicators['sma_EoM']=eom.sma_ease_of_movement()

        ### Force Index
        indicators['force_index']=ta.volume.ForceIndexIndicator(close= df['close'], volume= df['volume'], window= 13, fillna= False).force_index()

        ### Money Flow Index
        indicators['money_flow_index']=ta.volume.MFIIndicator(high= df['high'], low= df['low'], close= df['close'], volume=df['volume'], window = 14, fillna = False).money_flow_index()

        ### Negative Volume
        indicators['negative_volume_index']=ta.volume.NegativeVolumeIndexIndicator(close= df['close'], volume=df['volume'], fillna= False).negative_volume_index()

        ### On Balance Volume Indicator
        indicators['on_balance_volume']=ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'], fillna = False).on_balance_volume()

        ### Volume Price Trend Indicator
        indicators['volume_price_trend']=ta.volume.VolumePriceTrendIndicator(close=df['close'], volume=df['volume'], fillna = False).volume_price_trend()

        ### Volume Weighted Average Price
        indicators['volume_weightd_average_price']= ta.volume.VolumeWeightedAveragePrice(high= df['high'], low=df['low'], close=df['close'], volume=df['volume'], window= 14, fillna= False).volume_weighted_average_price()-df['close']
        if timestamp==True:
            indicators['timestamp']=df['timestamp']
        return indicators
    indicators = pd.DataFrame()
    indicators = trends(price, indicators, timestamp=timestamp)
    indicators = momentum(price, indicators, timestamp=timestamp)
    indicators = volume(price, indicators, timestamp=timestamp)
    indicators = volatility(price, indicators, timestamp=timestamp)
    return indicators

def tsfresh_rolling_features(df: pd.DataFrame,
                             time_series_col: str = "Close",
                             window_size: int = 2500,
                             start_idx: int = 11000,
                             preset: str = "comprehensive",
                             exclude: list = None,
                             n_jobs: int = 0) -> pd.DataFrame:
    """
    Extract rolling-window time series features using tsfresh.

    This function generates sliding windows from a single-column time series
    and uses tsfresh to automatically compute statistical features for each window.
    It can optionally exclude specific features (e.g. linear_trend) that
    trigger errors in certain SciPy versions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe. The index is ideally a datetime index.
        Must contain the column specified by `time_series_col`.
    time_series_col : str, default "Close"
        Column name containing the time series values (e.g. closing prices).
    window_size : int, default 2500
        The number of observations included in each rolling window.
    start_idx : int, default 11000
        The starting index for window extraction.
        This controls how many total samples will be generated.
        The function ensures it is at least `window_size - 1`.
    preset : {"comprehensive", "efficient", "minimal"}, default "comprehensive"
        The feature set template used by tsfresh.
        - "comprehensive": full set of hundreds of features (slowest)
        - "efficient": balanced speed vs. coverage
        - "minimal": smallest and fastest feature set
    exclude : list, optional
        A list of feature calculator names to remove from extraction.
        Defaults to removing ["linear_trend", "agg_linear_trend"]
        to avoid SciPy linregress-related NameError.
    n_jobs : int, default 0
        Number of parallel processes used by tsfresh.
        Set to -1 to use all available CPU cores.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame of extracted features.
        - Each row corresponds to one window (identified by its ending date).
        - Each column is a computed feature.
    """
    # Ensure the target column exists
    if time_series_col not in df.columns:
        raise KeyError(f"`{time_series_col}` not found in df.columns")

    # Select tsfresh feature extraction preset
    preset = preset.lower()
    if preset == "comprehensive":
        fc = ComprehensiveFCParameters()
    elif preset == "efficient":
        fc = EfficientFCParameters()
    elif preset == "minimal":
        fc = MinimalFCParameters()
    else:
        raise ValueError("preset must be one of {'comprehensive','efficient','minimal'}")

    # Exclude problematic or unwanted feature calculators
    if exclude is None:
        exclude = ["linear_trend", "agg_linear_trend"]
    for k in exclude:
        fc.pop(k, None)

    # Sort the series by index and select the target column
    s = df[[time_series_col]].sort_index()[time_series_col]

    # Check minimum length
    if len(s) < window_size:
        raise ValueError(f"Series length {len(s)} < window_size {window_size}.")

    # Ensure the starting point is valid
    start = max(start_idx, window_size - 1)

    # Create rolling windows
    windows = []
    for i in range(start, len(s)):
        win = s.iloc[i - window_size + 1 : i + 1]
        if win.isna().any():
            continue
        tmp = pd.DataFrame({
            "time": np.arange(window_size, dtype=int),  # time index within window
            "value": win.values,                        # window values
            "id": [s.index[i]] * window_size            # use end date as window ID
        })
        windows.append(tmp)

    # Validate at least one window created
    if not windows:
        raise RuntimeError("No valid windows were created. Check NaNs and indices.")

    # Concatenate all windows into a long-format dataframe
    full_df = pd.concat(windows, ignore_index=True)

    # Run tsfresh feature extraction
    feats = extract_features(
        full_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc,
        n_jobs=n_jobs
    )

    # Convert index to datetime if possible
    try:
        feats.index = pd.to_datetime(feats.index)
    finally:
        feats = feats.sort_index()

    return feats

def getEvent(price, threshold = 0.01):
  df=price.copy()
  df['Event']= peak_valley_pivots(df['close'], threshold, -threshold)
  target = df['Event'].shift(-1)  # Making target a lead
  return target

def compute_ATR(df, window=14, method="ema"):
    data = df.copy()
    data['prev_close'] = data['close'].shift(1)
    # True Range
    tr1 = data['high'] - data['low']
    tr2 = (data['high'] - data['prev_close']).abs()
    tr3 = (data['low'] - data['prev_close']).abs()
    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR
    if method == "sma":
        ATR = TR.rolling(window=window).mean()
    elif method == "ema":
        ATR = TR.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")
    return ATR


 #%%%
def get_full_dataset(data, threshold: float = 0.01):
    """
    Construct the full dataset by combining features, shifted event labels,
    and raw price information.

    Parameters
    ----------
    data : pd.DataFrame
        Input time series data that must contain at least the columns
        'close', 'low', and 'high'.
    threshold : float, optional (default=0.01)
        Threshold value used to generate event labels.

    Returns
    -------
    pd.DataFrame
        DataFrame that includes:
        - Engineered features from `getIndicators`
        - Event labels from `getEvent` (already shifted by -1, meaning each label
          corresponds to the event at the *next* time step)
        - Original price series: 'price', 'low', 'high'
        All rows with missing values are dropped.
    """
    # Compute feature indicators from input data
    together = getIndicators(data.copy(), False)

    # Compute event labels (shifted by -1 inside getEvent, so they represent next step events)
    event = getEvent(data.copy(), threshold)

    # Add shifted event labels to the feature set
    together["Event"] = event.values[:]

    # Attach raw price data
    together["price"] = data["close"].values[:]
    together["low"] = data["low"].values[:]
    together["high"] = data["high"].values[:]
    
    together["ATR"] = compute_ATR(data.copy(),window = 20, method = "ema")
    
    returns = together['price'].pct_change()
    
    together[f"ret_skew_{20}"] = returns.rolling(20).skew()

    # Remove rows with na values
    together.dropna(inplace=True)

    return together


'''def search_thresholds_global(
    df: pd.DataFrame,
    feat_cols: list,
    thresholds: list,
    n_splits: int = 4,
    embargo: int = 30,
    top_k: int = 20,
    shap_sample_size: int = 200,
    min_train: int = 500,
    w_auc: float = 0.3,
    w_conc: float = 0.4,
    w_stab: float = 0.3
):
    """
    Grid-search thresholds on the full dataset and pick the best one by a composite score.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (features + price/labels can be derived inside scoring).
    feat_cols : list[str]
        Candidate feature columns used by the RF in the scoring routine.
    thresholds : list[float]
        The zigzag/event thresholds to evaluate.
    n_splits : int
        Number of time-based folds in score_threshold_with_shap.
    embargo : int
        Embargo gap (number of samples) between train/validation in each fold.
    top_k : int
        Number of top features for SHAP concentration.
    shap_sample_size : int
        Max rows used for SHAP explanation in each fold.
    min_train : int
        Minimal number of labeled samples required; below this scoring may be skipped or penalized.
    w_auc, w_conc, w_stab : float
        Weights for the composite score: AUC, SHAP concentration, SHAP stability (Jaccard).

    Returns
    -------
    dict
        {
          "best_threshold": float,
          "best_score": float,
          "best_diag": dict,     # diagnostics for the best threshold
          "report": pd.DataFrame # per-threshold score & diagnostics, sorted desc by score
        }
    """
    
    def score_threshold_with_shap(
        df: pd.DataFrame,
        feat_cols: list,
        threshold: float,
        n_splits: int = 4,
        embargo: int = 30,
        top_k: int = 20,
        shap_sample_size: int = 200,
        min_train: int = 500,
        w_auc: float = 0.3,
        w_conc: float = 0.4,
        w_stab: float = 0.3
    ):
        """
        Score a fixed zigzag threshold by time-series CV with three criteria:
          - AUC (predictability): ROC-AUC on validation folds.
          - SHAP Concentration: share of total |SHAP| mass captured by top-K features.
          - SHAP Stability: Jaccard similarity of top-K sets across folds.

        Returns
        -------
        score : float
            Weighted sum: w_auc * mean(AUC) + w_conc * mean(concentration) + w_stab * stability.
        diag : dict
            Diagnostics with per-metric means.
        """
        
        def label_with_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:

            tmp = df.copy()
            events = getEvent(tmp, threshold)
            out = df.copy()
            out["Event"] = events.values[:]
            return out
        
        # 1) Apply your labeling at this threshold
        df_lab = label_with_threshold(df, threshold)

        # Keep only directional labels (-1/1); this makes it a binary classification
        df_lab = df_lab[df_lab["Event"].isin([-1, 1])].copy()
        if len(df_lab) < min_train:
            return -1e18, {"msg": "too_few_labeled"}

        N = len(df_lab)
        fold = N // (n_splits + 1)
        aucs, concs, topk_sets = [], [], []

        for i in range(n_splits):
            end_tr = fold * (i + 1)
            beg_te = end_tr + embargo
            end_te = min(fold * (i + 2), N)
            if end_te <= beg_te or end_te > N:
                continue

            tr = df_lab.iloc[:end_tr]
            va = df_lab.iloc[beg_te:end_te]
            if len(tr) < 200 or len(va) < 100:
                continue

            # 2) Train a small RF on the fold
            Xtr = tr[feat_cols]
            ytr = (tr["Event"] == 1).astype(int)  # 1 vs 0
            Xva = va[feat_cols]
            yva = (va["Event"] == 1).astype(int)

            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(Xtr, ytr)

            # 3) AUC on validation; robust to degenerate cases
            try:
                proba = rf.predict_proba(Xva)[:, 1]
                auc = roc_auc_score(yva, proba)
            except Exception:
                auc = 0.5
            aucs.append(auc)

            # 4) SHAP on the TRAIN fold only
            sample_n = min(shap_sample_size, len(Xtr))
            sample_X = Xtr.sample(sample_n, random_state=42)

            expl = shap.TreeExplainer(rf)
            shap_vals = expl.shap_values(sample_X)  # list of arrays, one per class: (n_samples, n_features)

            # Average |SHAP| over classes -> (n_features,n_samples),
            # then average over samples to get per-feature importance -> (n_features,)
            abs_mean_over_classes = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
            mean_abs_shap = abs_mean_over_classes.mean(axis=1)

            total = mean_abs_shap.sum() + 1e-12
            top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
            conc = float(mean_abs_shap[top_idx].sum() / total)  # concentration of top-K
            concs.append(conc)

            # Store the names of top-K features to measure stability across folds
            topk_sets.append(set(np.array(feat_cols)[top_idx]))

        # 5) SHAP stability across folds (mean Jaccard)
        stab = 0.0
        if len(topk_sets) >= 2:
            pair = 0
            jac = 0.0
            for i in range(len(topk_sets)):
                for j in range(i + 1, len(topk_sets)):
                    inter = len(topk_sets[i] & topk_sets[j])
                    union = len(topk_sets[i] | topk_sets[j])
                    jac += inter / (union + 1e-12)
                    pair += 1
            stab = jac / pair

        if not aucs or not concs:
            return -1e18, {"msg": "no_valid_folds"}

        # 6) Aggregate score (weights are user-tunable)
        auc_m  = float(np.mean(aucs))
        conc_m = float(np.mean(concs))

        score = w_auc * auc_m + w_conc * conc_m + w_stab * stab

        diag = {
            "auc_mean": auc_m,
            "shap_concentration_mean": conc_m,
            "shap_stability_jaccard": stab,
        }
        return score, diag
    
    rows = []
    best_thr, best_score, best_diag = None, -1e18, {}

    for thr in thresholds:
        s, d = score_threshold_with_shap(
            df=df,
            feat_cols=feat_cols,
            threshold=thr,
            n_splits=n_splits,
            embargo=embargo,
            top_k=top_k,
            shap_sample_size=shap_sample_size,
            min_train=min_train,
            w_auc=w_auc,
            w_conc=w_conc,
            w_stab=w_stab,
        )

        row = {"threshold": thr, "score": s}
        row.update(d)   # add diagnostics like auc_mean, shap_concentration_mean, shap_stability_jaccard, etc.
        rows.append(row)

        if s > best_score:
            best_thr, best_score, best_diag = thr, s, d

    report = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "best_threshold": best_thr,
        "best_score": best_score,
        "best_diag": best_diag,
        "report": report
    }

find_threshold = getIndicators(data.copy(), False)
find_threshold["close"] = find_threshold["close"].values[:]
find_threshold["low"] = data["low"].values[:]
find_threshold["high"] = data["high"].values[:]
find_threshold.dropna(inplace=True)

drop_cols = {"close","low","high"}
feat_cols = [c for c in find_threshold.columns if c not in drop_cols]

candidates = np.linspace(0.005, 0.03, 25)
res = search_thresholds_global(
    find_threshold,
    feat_cols=feat_cols,
    thresholds=candidates,
    n_splits=4,
    embargo=30,
    top_k=20,
    shap_sample_size=200,
    min_train=500,
    w_auc = 0.3,
    w_conc = 0.4,
    w_stab = 0.3
)
 
print("Best threshold:", res["best_threshold"])


print(res["report"].head(10))

thr = res["best_threshold"]'''


time_series_features_df = tsfresh_rolling_features(
    data.copy(), time_series_col="close",
    window_size=2500, start_idx=2500,
    preset="comprehensive",
    n_jobs=0
)

together = get_full_dataset(data.copy(),0.005)

together = pd.merge(
    time_series_features_df,
    together,
    left_index=True,     
    right_index=True,
    how='inner'           
)

data = pd.read_parquet('data.parquet')
higher_frequency_data = pd.read_parquet('higher_frequency_data.parquet')
together = pd.read_parquet('together.parquet')


features = getIndicators(data.copy(), False)
features.dropna(inplace=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


# =========================
# Step 0) 基础工具函数
# =========================
def keep_numeric_nonconstant(X: pd.DataFrame) -> pd.DataFrame:
    """只保留数值列，并去掉常数列（方差为0）"""
    Xn = X.select_dtypes(include=[np.number]).copy()
    nunique = Xn.nunique(dropna=False)
    Xn = Xn.loc[:, nunique > 1]
    return Xn


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个特征的 VIF。
    注意：内部会 dropna（只在 complete-case 上算 VIF）
    """
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
    """
    Step 1) VIF 迭代剔除：每次删掉 VIF 最大的特征，直到全部 <= vif_thresh
    返回：X_pruned, removed_list, final_vif_table
    """
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

        # VIF 可能 inf / nan：直接删掉最大那个继续
        if np.isnan(top_vif) or np.isinf(top_vif) or top_vif > vif_thresh:
            Xc = Xc.drop(columns=[top_feat])
            removed.append(top_feat)
        else:
            break

    final_vif = compute_vif(Xc) if Xc.shape[1] >= 2 else pd.DataFrame({"feature": Xc.columns, "vif": [1.0]*Xc.shape[1]})
    return Xc, removed, final_vif


def mi_feature_target(X: pd.DataFrame, y: np.ndarray, task: str = "regression",
                      n_neighbors: int = 5, random_state: int = 42) -> pd.Series:
    """
    计算每个 feature 与 target 的 MI：MI(X_i ; y)
    用于决定冗余 pair 中保留哪个（保留对 y 更相关的）。
    """
    Xn = keep_numeric_nonconstant(X)
    # 对齐 y 与 X 的 index（假设 y 是与 X 同长度的 ndarray/Series）
    y_arr = np.asarray(y)

    # 对 MI 来说最好不要有 NaN：这里取 complete-case
    mask = Xn.notna().all(axis=1) & ~pd.isna(y_arr)
    Xc = Xn.loc[mask]
    yc = y_arr[mask]

    if Xc.shape[0] == 0:
        return pd.Series(0.0, index=Xn.columns)

    if task == "classification":
        mi = mutual_info_classif(Xc, yc, discrete_features=False,
                                 n_neighbors=n_neighbors, random_state=random_state)
    else:
        mi = mutual_info_regression(Xc, yc, discrete_features=False,
                                    n_neighbors=n_neighbors, random_state=random_state)

    return pd.Series(mi, index=Xc.columns).reindex(Xn.columns).fillna(0.0)


def pairwise_mi_symmetric(X: pd.DataFrame, n_neighbors: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    计算特征两两之间的 MI 矩阵（连续变量）。
    MI 不天然对称（因为 sklearn 是用回归方式估计），所以用 (MI(i->j)+MI(j->i))/2 做对称化。
    """
    Xn = keep_numeric_nonconstant(X)
    Xc = Xn.dropna(axis=0)  # complete-case
    cols = list(Xc.columns)
    n = len(cols)

    if n == 0:
        return pd.DataFrame()
    if n == 1:
        return pd.DataFrame([[0.0]], index=cols, columns=cols)

    arr = Xc.to_numpy()
    M = np.zeros((n, n), dtype=float)

    def _mi_1d(x, y):
        # y 连续，用 mutual_info_regression
        return float(mutual_info_regression(x.reshape(-1, 1), y,
                                            discrete_features=False,
                                            n_neighbors=n_neighbors,
                                            random_state=random_state)[0])

    for i in range(n):
        for j in range(i + 1, n):
            mi_ij = _mi_1d(arr[:, i], arr[:, j])
            mi_ji = _mi_1d(arr[:, j], arr[:, i])
            mij = 0.5 * (mi_ij + mi_ji)
            M[i, j] = mij
            M[j, i] = mij

    return pd.DataFrame(M, index=cols, columns=cols)


def corr_mi_prune(X: pd.DataFrame,
                  corr_thresh: float = 0.90,
                  mi_thresh: float | None = None,
                  y: np.ndarray | None = None,
                  task: str = "regression",
                  n_neighbors: int = 5,
                  random_state: int = 42):
    """
    Step 2) 组内去冗余：
    - 若 |corr| >= corr_thresh -> 认为冗余
    - 若 mi_thresh 给定且 MI >= mi_thresh -> 认为冗余（非线性冗余）

    冗余 pair (a,b) 的删法：
    - 如果给了 y：删掉 MI(a;y) 更小的那个（保留更“有用”的）
    - 如果没给 y：删掉方差更小的那个（保守兜底）

    返回：X_pruned, removed_list
    """
    Xn = keep_numeric_nonconstant(X).copy()
    if Xn.shape[1] <= 1:
        return Xn, []

    # 用 complete-case 计算 corr / MI（避免 NaN）
    Xc = Xn.dropna(axis=0)
    if Xc.shape[1] <= 1 or Xc.shape[0] < 5:
        return Xn, []

    corr = Xc.corr().abs()
    mi_mat = pairwise_mi_symmetric(Xc, n_neighbors=n_neighbors, random_state=random_state) if mi_thresh is not None else None
    var = Xc.var()

    # 如果有 y，先算每个特征对 y 的 MI，用于 tie-break
    rel = None
    if y is not None:
        rel = mi_feature_target(Xn, y, task=task, n_neighbors=n_neighbors, random_state=random_state)

    cols = list(Xc.columns)
    pairs = []

    # 收集冗余 pairs
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = float(corr.loc[a, b])
            m = float(mi_mat.loc[a, b]) if mi_mat is not None else -np.inf

            if (c >= corr_thresh) or (mi_mat is not None and m >= mi_thresh):
                pairs.append((a, b, c, m))

    # 按严重程度排序：先看 corr，再看 MI
    pairs.sort(key=lambda t: (t[2], t[3]), reverse=True)

    keep = set(Xn.columns)
    removed = []

    def choose_drop(a, b):
        if rel is not None:
            return a if rel[a] < rel[b] else b
        return a if var.get(a, 0.0) < var.get(b, 0.0) else b

    for a, b, c, m in pairs:
        if a not in keep or b not in keep:
            continue
        d = choose_drop(a, b)
        keep.remove(d)
        removed.append(d)

    Xout = Xn.loc[:, [c for c in Xn.columns if c in keep]]
    return Xout, removed


def fit_group_pca(X: pd.DataFrame, cols: list[str], standardize: bool = True, n_components: int = 1):
    """
    Step 3) 在训练集上拟合 组内 PCA（targeted PCA）
    返回：scaler(可为None), pca, used_cols
    """
    Xg = X[cols].copy()
    Xg = Xg.dropna(axis=0)  # PCA 训练最好 complete-case
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


def transform_group_pca(X: pd.DataFrame, cols: list[str], scaler, pca, out_prefix: str = "group", n_components: int = 1):
    """
    用训练拟合好的 scaler/pca 在任意数据集上生成 PC 特征。
    遇到缺失行：输出 NaN（不会偷偷用不完整数据）
    """
    Xg = X[cols].copy()
    mask = Xg.notna().all(axis=1)

    Zout = np.full((X.shape[0], n_components), np.nan, dtype=float)
    if scaler is None or pca is None:
        # PCA没拟合成功：直接返回空 DF
        return pd.DataFrame(index=X.index)

    Xcomplete = Xg.loc[mask]
    if len(Xcomplete) > 0:
        Z = Xcomplete.to_numpy()
        if scaler is not None:
            Z = scaler.transform(Z)
        Zout[mask.values, :] = pca.transform(Z)

    pc_cols = [f"{out_prefix}_pc{i+1}" for i in range(n_components)]
    return pd.DataFrame(Zout, index=X.index, columns=pc_cols)


# =========================
# 你真正要跑的主流程（train fit -> transform train/val/test）
# =========================

# 例子：你定义每个“经济含义组”的特征列
groups = {
    "vol":   ["atr14", "atr21", "rv_5d", "rv_10d", "garch_sigma"],
    "trend": ["ma_slope_20", "adx_14", "mom_10", "mom_20"],
    "regime":["vix_level", "vix_term", "skew_proxy"],
}

# 你的阈值（建议先这样起步，再调）
VIF_THRESH  = 10.0
CORR_THRESH = 0.90
MI_THRESH   = 0.05     # MI 没有统一尺度，这个需要你按数据调
TASK        = "regression"  # 或 "classification"

MI_NEIGHBORS = 5
RANDOM_STATE = 42

PCA_STANDARDIZE = True
PCA_NCOMP = 1

# 假设你已经有：
# X_train: pd.DataFrame, y_train: np.ndarray or pd.Series
# X_val, X_test: pd.DataFrame

def run_pipeline_fit_transform(X_train, y_train, X_list_to_transform: list[pd.DataFrame]):
    """
    在 X_train 上：
      - 每组：VIF prune -> corr/MI prune -> fit PCA
    然后把同样的处理（用 train 学到的选择与 PCA 参数）应用到 val/test。
    返回：
      - X_train_out
      - list of X_out for X_list_to_transform
      - 日志 dict（每组删了什么、保留了什么、PCA解释方差）
    """
    X_train_num = keep_numeric_nonconstant(X_train)

    logs = {
        "vif_removed": {},
        "corr_mi_removed": {},
        "kept_cols": {},
        "pca_explained_var": {}
    }

    # 保存：每组最终用于 PCA 的列 + scaler/pca
    fitted = {}

    # ---- (A) 在 train 上对每个组做 VIF + corr/MI + PCA 拟合
    for gname, cols in groups.items():
        cols_present = [c for c in cols if c in X_train_num.columns]
        if len(cols_present) == 0:
            continue

        # Step 1) VIF prune（只在该组内）
        Xg0 = X_train_num[cols_present]
        Xg1, removed_vif, final_vif = vif_prune_iterative(Xg0, vif_thresh=VIF_THRESH)
        logs["vif_removed"][gname] = removed_vif

        # Step 2) corr + MI prune（只在该组内）
        Xg2, removed_cm = corr_mi_prune(
            Xg1,
            corr_thresh=CORR_THRESH,
            mi_thresh=MI_THRESH,
            y=y_train,               # 有 y 更合理
            task=TASK,
            n_neighbors=MI_NEIGHBORS,
            random_state=RANDOM_STATE
        )
        logs["corr_mi_removed"][gname] = removed_cm

        kept = list(Xg2.columns)
        logs["kept_cols"][gname] = kept

        # Step 3) fit PCA（只在 kept 上）
        scaler, pca, used_cols = fit_group_pca(
            X_train_num, kept,
            standardize=PCA_STANDARDIZE,
            n_components=PCA_NCOMP
        )

        if pca is not None:
            logs["pca_explained_var"][gname] = float(pca.explained_variance_ratio_[0])
            fitted[gname] = {"cols": used_cols, "scaler": scaler, "pca": pca}
        else:
            # kept <2 或数据不足：不做 PCA，后面就直接保留 kept 原特征
            fitted[gname] = {"cols": used_cols, "scaler": None, "pca": None}

    # ---- (B) 组外特征：如果你要保留不在任何组里的列
    all_group_cols = set([c for cols in groups.values() for c in cols])
    nongroup_cols = [c for c in X_train_num.columns if c not in all_group_cols]

    # ---- (C) 生成 train 输出：non-group + (每组 pc1 或 kept)
    out_parts_train = []
    if len(nongroup_cols) > 0:
        out_parts_train.append(X_train_num[nongroup_cols])

    for gname, obj in fitted.items():
        cols_used = obj["cols"]
        if obj["pca"] is not None:
            pc = transform_group_pca(
                X_train_num, cols_used,
                obj["scaler"], obj["pca"],
                out_prefix=gname,
                n_components=PCA_NCOMP
            )
            out_parts_train.append(pc)
            # 你说的 targeted PCA 通常会丢掉原始组内特征，所以这里不再 append 原始 cols_used
        else:
            # 没PCA就直接保留 kept features
            out_parts_train.append(X_train_num[cols_used])

    X_train_out = pd.concat(out_parts_train, axis=1)

    # ---- (D) transform val/test：用 train 的 fitted 结果
    X_out_list = []
    for X_any in X_list_to_transform:
        Xn = keep_numeric_nonconstant(X_any)

        parts = []
        if len(nongroup_cols) > 0:
            # 只保留这些列中在 X_any 里存在的
            cols_exist = [c for c in nongroup_cols if c in Xn.columns]
            if len(cols_exist) > 0:
                parts.append(Xn[cols_exist])

        for gname, obj in fitted.items():
            cols_used = [c for c in obj["cols"] if c in Xn.columns]
            if len(cols_used) == 0:
                continue

            if obj["pca"] is not None and len(cols_used) == len(obj["cols"]):
                pc = transform_group_pca(
                    Xn, cols_used,
                    obj["scaler"], obj["pca"],
                    out_prefix=gname,
                    n_components=PCA_NCOMP
                )
                parts.append(pc)
            else:
                # 如果 test/val 缺列，或该组本来没 PCA：退化为保留可用列
                parts.append(Xn[cols_used])

        X_out_list.append(pd.concat(parts, axis=1) if len(parts) else pd.DataFrame(index=X_any.index))

    return X_train_out, X_out_list, logs


X_train_out, [X_val_out, X_test_out], logs = run_pipeline_fit_transform(X_train, y_train, [X_val, X_test])
print(logs["vif_removed"])
print(logs["corr_mi_removed"])
print(logs["pca_explained_var"])




 #%%%
together['Event'].value_counts()

#%%
def embargo_split(X, test_proportion: float, embargo_period: int):
    """
    Split a dataset into training and testing sets with an embargo period.

    The embargo period removes a buffer zone of observations between
    the training and test sets to prevent information leakage.

    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Input dataset to be split.
    test_proportion : float
        Fraction of the dataset to allocate to the test set (e.g., 0.2 for 20%).
    embargo_period : int
        Number of samples to exclude between training and test sets.

    Returns
    -------
    (train, test) : tuple
        - train : pd.DataFrame or pd.Series, training subset
        - test  : pd.DataFrame or pd.Series, testing subset after embargo
    """
    # Calculate test set size based on proportion
    size = int(len(X) * test_proportion)

    # Training set size = total length - test size
    train_size = len(X) - size

    return X.iloc[:train_size], X.iloc[train_size + embargo_period:]

def plot_feature_hist(x, title: str):
    """
    Plot a histogram of a feature along with a normal distribution fit
    and a kernel density estimate (KDE).

    This function provides a quick visual check of the feature's distribution
    by showing:
    - Histogram of the data
    - Fitted normal distribution curve
    - KDE curve (non-parametric estimate)

    Parameters
    ----------
    x : array-like
        Input feature values to visualize.
    title : str
        Title for the plot (also used as the x-axis label).

    Returns
    -------
    None
        Displays a matplotlib figure.
    """
    # Ensure input is a NumPy array
    x = np.asarray(x)

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot histogram of the feature
    plt.hist(
        x, bins=60, density=True, alpha=0.6,
        color="skyblue", edgecolor="black", label="Histogram"
    )

    # Fit and plot normal distribution curve
    mu, sigma = np.mean(x), np.std(x)
    xx = np.linspace(x.min(), x.max(), 400)
    plt.plot(
        xx, stats.norm.pdf(xx, mu, sigma),
        'r', linewidth=2,
        label=f'Normal fit (μ={mu:.4g}, σ={sigma:.4g})'
    )

    # Fit and plot KDE curve
    kde = stats.gaussian_kde(x)
    plt.plot(xx, kde(xx), 'g', linewidth=2, label="KDE fit")

    # Add labels and decorations
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def RF_meta_label_model(
    base_reg_df: pd.DataFrame,
    meta_reg_df: pd.DataFrame,
    top_k_shap: int,
    feat_cols: list,
    shap_sample_size: int = 100,
    alpha: float = 0.5
):
    """
    Train a base direction model (Random Forest) inside one regime, select top-K features via SHAP,
    apply conformal-like screening to keep only high-confidence predictions (others -> 0/hold),
    then train a meta model that predicts whether the base prediction will be correct.

    Parameters
    ----------
    base_reg_df : pd.DataFrame
        Regime subset used to train the base model. Must contain 'Event' and feature columns.
    meta_reg_df : pd.DataFrame
        Regime subset used to train the meta model. Must contain 'Event' and feature columns.
    top_k_shap : int
        Number of top features to keep based on SHAP importance.
    feat_cols : list[str]
        Candidate feature columns (label/price/etc. already excluded).
    shap_sample_size : int, default 100
        Max rows sampled for SHAP explanations; actual sample size is min(sample_size, len(train_bin)).
    alpha : float, default 0.5
        Conformal selection parameter in (0,1). Larger alpha -> stricter (keep higher-confidence only).

    Returns
    -------
    meta_model : RandomForestClassifier
    base_model : RandomForestClassifier
    threshold  : float
    top_k_features : list[str]
    """
    
    
    # Use only labeled directions (-1/1) to train the base model
    train_bin = base_reg_df[base_reg_df["Event"].isin([-1, 1])].copy()
    if len(train_bin) == 0:
        raise ValueError("No labeled samples (-1/1) in base_reg_df for training.")

    # Train a RF to compute SHAP importance
    reducing_feat_model = RandomForestClassifier(n_estimators=1000, random_state=42)
    reducing_feat_model.fit(train_bin[feat_cols], train_bin["Event"])

    # SHAP: sample at most 'shap_sample_size' rows (and >=1)
    sample_n = min(shap_sample_size, len(train_bin))
    sample_X = train_bin[feat_cols].sample(sample_n, random_state=42)

    explainer = shap.TreeExplainer(reducing_feat_model)
    shap_values = explainer.shap_values(sample_X)  # list[classes] of (n_features, n_samples)

    # Mean absolute SHAP per feature (average over classes, then over samples)
    
    mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=1)
    shap_ranking = pd.Series(mean_abs_shap, index=train_bin[feat_cols].columns).sort_values(ascending=False)
    top_k_features = shap_ranking.head(top_k_shap).index.tolist()

    # Train the base model on top-K features and produce predictions
    base_model = RandomForestClassifier(n_estimators=200, random_state=4)
    base_model.fit(train_bin[top_k_features], train_bin["Event"])

    base_preds_meta = base_model.predict(meta_reg_df[top_k_features])
    base_probs_meta = base_model.predict_proba(meta_reg_df[top_k_features]).max(axis=1)

    # Conformal-like threshold: 1 - max_proba on base set -> take (1 - alpha) quantile
    # conformal score is like score for uncertainty
    # threshold is the threshold for uncertainty, the conformal score of signal less than the threshold will be replaced by 0
    probs_base = base_model.predict_proba(base_reg_df[top_k_features])
    max_probs_base = probs_base.max(axis=1)
    conformal_score = 1 - max_probs_base
    threshold = np.quantile(conformal_score, 1 - alpha)

    # Apply on meta set: keep if prob >= 1 - threshold, else 0 (hold)
    adjusted_preds = np.where(base_probs_meta >= 1 - threshold, base_preds_meta, 0)

    # Train the meta model to predict correctness of base predictions
    meta_remain = meta_reg_df[top_k_features + ["Event"]].copy()
    meta_remain["signals"] = adjusted_preds
    meta_remain["meta_feature"] = np.where(meta_remain["signals"] == meta_remain["Event"], 1, 0)

    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(meta_remain[top_k_features], meta_remain["meta_feature"])
    
    

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
    skew_col: str = "ret_skew_20",   # column name of rolling skewness
    skew_threshold: float = 0.5      # if high-vol regime & skew > this → no trade
):
    """
    End-to-end pipeline using:
      - HMM for volatility regime detection,
      - Global RF base + meta model (NOT per-regime),
      - HMM + skewness as a trading condition filter on the test set.

    Steps
    -----
    1) Ensure train/test have 'returns' and 'log_ret2'.
    2) Embargo split train_df into base_model_df and meta_model_df (for RF/meta training).
    3) Fit HMM on the ENTIRE train_df (log_ret2) to learn volatility regimes.
    4) Use HMM to infer regimes for base_model_df, meta_model_df, and test_df.
    5) Rank regimes by average volatility (log_ret2 mean) and identify the high-vol regime.
    6) Train a single global RF + meta-label model using base/meta (across all regimes).
    7) On test_df:
         - Get base predictions + conformal filtering → signals_raw
         - Apply trading condition:
             if (regime == high-vol) and (skew > skew_threshold) → final signal = 0 (no trade)
    8) Plot confusion matrix using final signals.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set (already embargoed from test; includes features, Event, price, etc.).
    test_df : pd.DataFrame
        Test set.
    embargo_period : int
        Embargo window when splitting train_df into base/meta.
    n_components, n_mix, hmm_iter, hmm_seed : HMM hyperparameters.
    top_k_shap, shap_sample_size, alpha : RF/SHAP/meta parameters.
    skew_col : str
        Column name for rolling skewness used in the trading condition.
    skew_threshold : float
        Threshold for skewness; if skew > skew_threshold in high-vol regime, we do not trade.

    Returns
    -------
    artifacts : dict
        {
          "base_model_df": pd.DataFrame (with regimes),
          "meta_model_df": pd.DataFrame (with regimes),
          "test_df": pd.DataFrame       (with regimes),
          "models": {
              "base": base_model,
              "meta": meta_model,
              "threshold": conformal_threshold,
              "features": top_k_features,
              "hmm_model": hmm_model,
              "high_vol_regime": int,
              "low_vol_regime": int,
              "mid_vol_regime": Optional[int],
          },
          "testset": pd.DataFrame (test with signals, final signals, meta_label_probs, regimes),
        }
    """

    def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the DataFrame has basic return-based features:
          - 'returns'  : simple percentage returns based on 'price'
          - 'log_ret2' : log of squared returns (proxy for volatility)
        """
        df = df.copy()
        if "returns" not in df:
            df["returns"] = df["price"].pct_change()
        if "log_ret2" not in df:
            df["log_ret2"] = np.log(df["returns"] ** 2 + 1e-8)
        df.dropna(subset=["returns"], inplace=True)
        return df

    # 1) Make sure both train and test have 'returns' and 'log_ret2'
    train_df = _ensure_features(train_df)
    test_df  = _ensure_features(test_df)

    # 2) Embargo-split the TRAIN set into base/meta for RF/meta training
    base_model_df, meta_model_df = embargo_split(
        train_df.copy(), test_proportion=0.3, embargo_period=embargo_period
    )

    # 3) Fit HMM on the ENTIRE TRAIN SET (train_df), not just base_model_df
    returns_train = base_model_df[["returns"]].dropna().values
    hmm_model = GMMHMM(
        n_components=3,
        n_mix=4,
        covariance_type="diag",
        n_iter=300,
        random_state=200,
    ).fit(returns_train)

    # Helper: infer regimes and regime probabilities on any dataframe
    def _infer_regimes_on(df_part: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted HMM to assign a discrete regime label and
        per-regime probabilities to each row in df_part.
        """
        x = df_part[["returns"]].values
        states = hmm_model.predict(x)

        out = df_part.copy()
        out["regime"] = states
        return out

    # 4) Infer regimes for base/meta/test using the HMM trained on the entire train_df
    base_model_df = _infer_regimes_on(base_model_df)
    meta_model_df = _infer_regimes_on(meta_model_df)
    test_df       = _infer_regimes_on(test_df)

    # 5) Identify which regime is low/mid/high volatility using mean log_ret2 on BASE subset
    regime_vol = (
        base_model_df.groupby("regime")["returns"]
        .std()
        .sort_values()  # ascending: lowest volatility first
    )

    def relabel_regimes(df: pd.DataFrame, regime_vol: pd.Series) -> pd.DataFrame:
        """
        Relabel numeric HMM regimes into string volatility labels
        based on sorted regime_vol (ascending: low -> high).

        regime_vol.index contains the original regime IDs.
        """

        # Identify which original regime ID is low / mid / high
        low_vol_regime  = regime_vol.index[0]
        mid_vol_regime  = regime_vol.index[1] if len(regime_vol) > 2 else None
        high_vol_regime = regime_vol.index[-1]

        # Build a mapping: original regime ID -> label string
        regime_label_map = {
            low_vol_regime: "low_vol_regime",
            high_vol_regime: "high_vol_regime",
        }
        if mid_vol_regime is not None:
            regime_label_map[mid_vol_regime] = "mid_vol_regime"

        out = df.copy()
        # Create a new column with string labels
        out["regime_relabel"] = out["regime"].map(regime_label_map)

        return out
    
    base_model_df = relabel_regimes(base_model_df, regime_vol)
    meta_model_df = relabel_regimes(meta_model_df, regime_vol)
    test_df = relabel_regimes(test_df, regime_vol)

    # 6) Build the feature list (global, not per-regime)
    drop_cols = {"Event", "price", "regime", "returns", "low", "high", "ATR","regime_relabel","log_ret2"}
    feat_cols = [
        c for c in train_df.columns
        if (c not in drop_cols and not c.startswith("regime_p"))
    ]

    # 7) Train a single global RF base + meta model (across all regimes)
    meta_model, base_model, threshold, top_k_features, reducing_feat_model, sample_X = RF_meta_label_model(
        base_reg_df=base_model_df,
        meta_reg_df=meta_model_df,
        top_k_shap=top_k_shap,
        feat_cols=feat_cols,
        shap_sample_size=shap_sample_size,
        alpha=alpha,
    )

    # 8) Apply the models to the test set
    use_cols = [c for c in top_k_features if c in test_df.columns]
    testset = test_df.copy()

    # Meta model predicts the probability that the base signal will be correct
    testset["meta_label_probs"] = meta_model.predict_proba(testset[use_cols])[:, 1]

    base_preds = base_model.predict(testset[use_cols])
    base_probs = base_model.predict_proba(testset[use_cols]).max(axis=1)

    # Raw signals based only on conformal threshold (no HMM/skew filter yet)
    testset["signals_raw"] = np.where(base_probs >= 1 - threshold, base_preds, 0)

    # 9) Apply the trading condition: high-vol regime + high skewness → no trade
    if skew_col not in testset.columns:
        raise KeyError(
            f"Column '{skew_col}' not found in testset. "
            f"Make sure you computed rolling skewness in your feature construction step."
        )

    # Condition: (high volatility regime) AND (skewness > threshold)
    cond_block = (testset["regime_relabel"] == "high_vol_regime") & (abs(testset[skew_col]) > skew_threshold)

    # Final signals: if condition is met → 0 (hold), else keep raw signals
    testset["signals"] = np.where(cond_block, 0, testset["signals_raw"])

    # 10) Sort index and plot confusion matrix using final signals
    testset.index = pd.to_datetime(testset.index)
    testset = testset.sort_index()

    LABELS = np.array([-1, 0, 1])
    cm = confusion_matrix(testset["Event"], testset["signals"], labels=LABELS)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (HOLD=0, HMM + Skewness Filter)")
    plt.show()

    models = {
        "base": base_model,
        "meta": meta_model,
        "threshold": threshold,
        "features": top_k_features,
        "hmm_model": hmm_model
    }

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
    alpha: float = 0.5
):
    train_df = train_df.copy()
    test_df  = test_df.copy()
    
    base_model_df, meta_model_df = embargo_split(
        train_df.copy(), test_proportion=0.3, embargo_period=embargo_period
    )
    
    drop_cols = {"Event","price","regime","returns","low","high","ATR","second_model_label"}
    feat_cols = [c for c in train_df.columns if (c not in drop_cols)]
    
    meta_m, base_m, thr, top_feats, reducing_feature_model, sample_X = RF_meta_label_model(
        base_model_df, meta_model_df, top_k_shap=top_k_shap, feat_cols=feat_cols,
        shap_sample_size=shap_sample_size, alpha=alpha
    )
    
    test_df["meta_label_probs"] = meta_m.predict_proba(test_df[top_feats])[:, 1]

    base_preds = base_m.predict(test_df[top_feats])
    base_probs = base_m.predict_proba(test_df[top_feats]).max(axis=1)
    test_df["signals"] = np.where(base_probs >= 1 - thr, base_preds, 0)
    
    accuracy = accuracy_score(test_df["Event"], test_df["signals"])
    print(f'Accuracy: {accuracy}')
    
    LABELS = np.array([-1, 0, 1])        
    cm = confusion_matrix(test_df["Event"], test_df["signals"], labels=LABELS)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (with HOLD=0)")
    plt.show()
    
    return test_df,reducing_feature_model, sample_X
         

def strategy_backtesting_trailing_sl(
    predictions,
    prices,   # pd.Series (DatetimeIndex)
    higher_frequency_data,   # DataFrame (DatetimeIndex)
    low,
    high,
    meta_probs,
    atr,
    sl_k=2.0,
    tp_k=5.0,
    prob_threshold=0.3   # ← Added parameter: only trade when meta_prob > this threshold
):
    """
    Backtest a trading strategy with dynamic position sizing and ATR-based trailing stop-loss / take-profit.

    Parameters
    ----------
    predictions : array-like
        Model-predicted trading signals for each bar:
        -  1 → Peak (open/scale SHORT)
        - -1 → Valley (open/scale LONG)
        -  0 → Neutral (no trade)
    prices : pd.Series
        Asset close price series (DatetimeIndex aligned with predictions).
    higher_frequency_data : pd.DataFrame
        Higher-frequency data (e.g. 5-min bars) used to determine which threshold (TP/SL) was hit first.
        Must contain columns ['open', 'high', 'low', 'close'] and have DatetimeIndex.
    low : array-like
        Low prices for each bar in the base timeframe.
    high : array-like
        High prices for each bar in the base timeframe.
    meta_probs : array-like
        Probabilities output by the meta model, used as dynamic position sizes (scaling factor).
        *A new trade or scale-in will only be triggered when meta_prob > prob_threshold.*
    atr : array-like
        Average True Range values for each bar (used for volatility-based TP/SL).
    sl_k : float, default=2.0
        Stop-loss multiplier relative to ATR.
        - Long: SL = price - sl_k * ATR (trailing, only moves up)
        - Short: SL = price + sl_k * ATR (trailing, only moves down)
    tp_k : float, default=5.0
        Take-profit multiplier relative to ATR.
        - Long: TP = price + tp_k * ATR (fixed at entry)
        - Short: TP = price - tp_k * ATR (fixed at entry)
    prob_threshold : float, default=0.3
        Minimum probability required to open or scale a position.

    Returns
    -------
    returns_series : pd.Series
        Equity curve (normalized to 1.0 at start) indexed by time.
    trade_df : pd.DataFrame
        Trade log containing all entries and exits with details:
        - 'time': timestamp of trade
        - 'trade': action (open/close/flip)
        - 'trade_price': execution price
        - 'average_price': current position’s average entry price
        - 'trade_position': size of this trade (scaled by meta_prob)
        - 'current_total_position': total net position after trade

    Notes
    -----
    - Position size is proportional to meta_probs.
    - Stop-loss (SL) is *trailing* — it moves only in a favorable direction.
    - Take-profit (TP) is *fixed* — locked at entry and never updated.
    - Equity is updated mark-to-market at every bar.
    - Added logic: trades are only executed when meta_prob > prob_threshold.
    """

    # --- convert inputs to arrays for performance ---
    predictions = np.asarray(predictions)
    meta_probs  = np.asarray(meta_probs)
    low  = np.asarray(low)
    high = np.asarray(high)
    atr  = np.asarray(atr)

    # --- Precompute candidate stop-loss / take-profit levels ---
    l_tp_prices = prices.values + tp_k * atr
    l_sl_prices = prices.values - sl_k * atr
    s_tp_prices = prices.values - tp_k * atr
    s_sl_prices = prices.values + sl_k * atr

    # --- Initialize trading state variables ---
    position   = 0.0
    entry_idx  = None
    entry_px   = None
    entry_ts   = None
    curr_sl    = None
    fixed_tp   = None

    # --- Portfolio tracking ---
    equity = np.ones(len(prices), dtype=float)
    available_funds = 1.0
    trade_log = []

    # --- Helpers for trailing exit logic ---
    def exit_long(l_sl_price, high_t1, low_t1, bar_start, bar_end, window):
        nonlocal curr_sl, fixed_tp
        candidate_sl = l_sl_price
        curr_sl = max(curr_sl, candidate_sl)
        sl, tp = curr_sl, fixed_tp
        hit_sl = (low_t1 <= sl)
        hit_tp = (high_t1 >= tp)
        if hit_sl and hit_tp:
            result, which, px, _ = resolve_with_lower_tf(bar_start, bar_end, "long", sl, tp, window)
            return result, px, which
        elif hit_sl:
            return True, sl, "SL"
        elif hit_tp:
            return True, tp, "TP"
        return False, None, None

    def exit_short(s_sl_price, high_t1, low_t1, bar_start, bar_end, window):
        nonlocal curr_sl, fixed_tp
        candidate_sl = s_sl_price
        curr_sl = min(curr_sl, candidate_sl)
        sl, tp = curr_sl, fixed_tp
        hit_sl = (high_t1 >= sl)
        hit_tp = (low_t1 <= tp)
        if hit_sl and hit_tp:
            result, which, px, _ = resolve_with_lower_tf(bar_start, bar_end, "short", sl, tp, window)
            return result, px, which
        elif hit_sl:
            return True, sl, "SL"
        elif hit_tp:
            return True, tp, "TP"
        return False, None, None

    def _resolve_micro_order(open_, high_, low_, close_, side, sl, tp, model="auto"):
        if side == "long":
            if open_ <= sl: return "SL", sl
            if open_ >= tp: return "TP", tp
        else:
            if open_ >= sl: return "SL", sl
            if open_ <= tp: return "TP", tp
        if model == "HL":
            return ("TP", tp) if side == "long" else ("SL", sl)
        elif model == "LH":
            return ("SL", sl) if side == "long" else ("TP", tp)
        else:
            bullish = close_ >= open_
            if side == "long":
                return ("SL", sl) if bullish else ("TP", tp)
            else:
                return ("TP", tp) if bullish else ("SL", sl)

    def resolve_with_lower_tf(hourly_bar_start, hourly_bar_end, side, sl, tp, window, micro_model="auto"):
        for ts, row in window.iterrows():
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            if side == "long":
                hit_sl = (l <= sl)
                hit_tp = (h >= tp)
            else:
                hit_sl = (h >= sl)
                hit_tp = (l <= tp)
            if hit_tp and not hit_sl:
                return True, "TP", tp, ts
            if hit_sl and not hit_tp:
                return True, "SL", sl, ts
            if not hit_tp and not hit_sl:
                continue
            which, px = _resolve_micro_order(o, h, l, c, side=side, sl=sl, tp=tp, model=micro_model)
            return True, which, px, ts
        return False, None, None, None

    # ==============================
    # Main backtesting loop
    # ==============================
    for t in range(1, len(predictions)):
        signal  = predictions[t - 1]
        prob    = float(meta_probs[t - 1])
        low_t1  = float(low[t])
        high_t1 = float(high[t])
        bar_start = prices.index[t]
        bar_end = bar_start + pd.Timedelta(hours=1)
        window = higher_frequency_data.loc[
            (higher_frequency_data.index >= bar_start) &
            (higher_frequency_data.index < bar_end)
        ]

        # --- Step 1: manage existing positions (TP/SL handling) ---
        if entry_px is not None:
            l_sl_price = l_sl_prices[t-1]
            s_sl_price = s_sl_prices[t-1]
            if position > 0:
                exit_flag, exit_price, exit_reason = exit_long(l_sl_price, high_t1, low_t1, bar_start, bar_end, window)
                if exit_flag:
                    ret = (exit_price - entry_px) / entry_px
                    available_funds *= (1 + ret * position)
                    equity[t] = available_funds
                    trade_log.append({
                        'time': prices.index[t],
                        'trade': f'close long {exit_reason}',
                        'trade_price': exit_price,
                        'average_price': entry_px,
                        'trade_position': abs(position),
                        'current_total_position': 0
                    })
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None
            elif position < 0:
                exit_flag, exit_price, exit_reason = exit_short(s_sl_price, high_t1, low_t1, bar_start, bar_end, window)
                if exit_flag:
                    ret = (entry_px - exit_price) / entry_px
                    available_funds *= (1 + ret * abs(position))
                    equity[t] = available_funds
                    trade_log.append({
                        'time': prices.index[t],
                        'trade': f'close short {exit_reason}',
                        'trade_price': exit_price,
                        'average_price': entry_px,
                        'trade_position': abs(position),
                        'current_total_position': 0
                    })
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

        # --- Step 2: new signal handling (flip/open/scale) ---
        if prob > prob_threshold:  # ← Only act when meta_prob exceeds threshold
            if signal == 1:  # Peak → open/scale SHORT
                if position > 0:  # close long before flipping
                    pnl = (prices.iloc[t] - entry_px) / entry_px * position
                    available_funds *= (1 + pnl)
                    trade_log.append({
                        'time': prices.index[t],
                        'trade': 'close long (flip)',
                        'trade_price': float(prices.iloc[t]),
                        'average_price': entry_px,
                        'trade_position': abs(position),
                        'current_total_position': 0
                    })
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

                # open or scale short
                if entry_px is None:
                    entry_idx = t
                    entry_px  = float(prices.iloc[t])
                    entry_ts  = prices.index[t]
                    fixed_tp  = float(s_tp_prices[entry_idx])
                    curr_sl   = float(s_sl_prices[entry_idx])
                else:
                    entry_px = (entry_px * abs(position) + float(prices.iloc[t]) * prob) / (abs(position) + prob)

                position -= prob
                trade_log.append({
                    'time': prices.index[t],
                    'trade': 'short',
                    'trade_price': float(prices.iloc[t]),
                    'average_price': entry_px,
                    'trade_position': abs(prob),
                    'current_total_position': position
                })

            elif signal == -1:  # Valley → open/scale LONG
                if position < 0:  # close short before flipping
                    pnl = (entry_px - prices.iloc[t]) / entry_px * abs(position)
                    available_funds *= (1 + pnl)
                    trade_log.append({
                        'time': prices.index[t],
                        'trade': 'close short (flip)',
                        'trade_price': float(prices.iloc[t]),
                        'average_price': entry_px,
                        'trade_position': abs(position),
                        'current_total_position': 0
                    })
                    position = 0.0
                    entry_idx = entry_px = entry_ts = None
                    curr_sl = fixed_tp = None

                # open or scale long
                if entry_px is None:
                    entry_idx = t
                    entry_px  = float(prices.iloc[t])
                    entry_ts  = prices.index[t]
                    fixed_tp  = float(l_tp_prices[entry_idx])
                    curr_sl   = float(l_sl_prices[entry_idx])
                else:
                    entry_px = (entry_px * abs(position) + float(prices.iloc[t]) * prob) / (abs(position) + prob)

                position += prob
                trade_log.append({
                    'time': prices.index[t],
                    'trade': 'long',
                    'trade_price': float(prices.iloc[t]),
                    'average_price': entry_px,
                    'trade_position': abs(prob),
                    'current_total_position': position
                })

        # --- Step 3: mark-to-market equity update ---
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
    return returns_series, trade_df

def strategy_backtesting_average_change_rate(
    predictions,
    prices,                   # pd.Series (DatetimeIndex)
    meta_probs,               # array-like, same length as prices
    train_set_price_event,    # DataFrame with columns ['event','price'] (or 'Event'/'Price')
    prob_threshold=0.3        # Only trade when meta_prob > threshold
):
    """
    Backtest a strategy using TP/SL thresholds derived from the average absolute
    peak↔valley return computed on the training set. 
    Position size is scaled by meta_probs, and trades are executed only when 
    meta_prob > prob_threshold.

    Returns
    -------
    returns_series : pd.Series
        Equity curve (starting from 1.0), indexed by prices.index.
    trade_df : pd.DataFrame
        Trade log with time, trade type, trade price, average price, 
        trade position, and total position after the trade.
    """

    predictions = np.asarray(predictions)
    meta_probs  = np.asarray(meta_probs)

    # --- Compute the average absolute return between peaks and valleys on the training set ---
    def compute_avg_pivot_change(df):
        evt_col = 'event' if 'event' in df.columns else ('Event' if 'Event' in df.columns else None)
        if evt_col is None:
            raise KeyError("train_set_price_event must contain 'event' or 'Event'.")
        px_col  = 'price' if 'price' in df.columns else ('Price' if 'Price' in df.columns else None)
        if px_col is None:
            raise KeyError("train_set_price_event must contain 'price' or 'Price'.")

        df = df.copy()
        df["event_shifted"] = df[evt_col].shift(1)

        change_rates = []
        last_pivot_price = None
        last_pivot_type  = None
        for _, row in df.iterrows():
            event, price = row["event_shifted"], row[px_col]
            if pd.notna(event) and event != 0:
                if last_pivot_price is not None and last_pivot_type != event:
                    # Compute absolute rate of change between two consecutive pivots of opposite type
                    rate = (price - last_pivot_price) / last_pivot_price
                    change_rates.append(abs(rate))
                last_pivot_price, last_pivot_type = price, event

        avg_rate = float(np.mean(change_rates)) if change_rates else 0.0
        print(f"Average peak-valley absolute return: {avg_rate:.4%}")
        return avg_rate

    avg_change = compute_avg_pivot_change(train_set_price_event)
    avg_change = max(avg_change, 1e-4)  # Avoid immediate liquidation if avg_change == 0

    # --- Initialize state variables ---
    position = 0.0               # >0 = long, <0 = short
    entry_px = None
    equity = np.ones(len(prices), dtype=float)
    available_funds = 1.0
    trade_log = []

    # --- Main backtesting loop ---
    for t in range(1, len(predictions)):
        signal = float(predictions[t - 1])
        prob   = float(meta_probs[t - 1])
        cur_px = float(prices.iloc[t])

        # Step 1: Manage existing positions (Take-Profit / Stop-Loss)
        if entry_px is not None:
            ret = (cur_px - entry_px) / entry_px  # Return since entry

            if position > 0:
                # Long position TP/SL
                if ret >= avg_change:  # Take-Profit
                    available_funds *= (1 + ret * position)
                    equity[t] = available_funds
                    trade_log.append({
                        "time": prices.index[t], "trade": "close long TP",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None
                elif ret <= -avg_change:  # Stop-Loss
                    available_funds *= (1 + ret * position)
                    equity[t] = available_funds
                    trade_log.append({
                        "time": prices.index[t], "trade": "close long SL",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None

            elif position < 0:
                # Short position TP/SL
                if ret <= -avg_change:  # Take-Profit (price falls)
                    available_funds *= (1 + ret * position)  # position < 0
                    equity[t] = available_funds
                    trade_log.append({
                        "time": prices.index[t], "trade": "close short TP",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None
                elif ret >= avg_change:  # Stop-Loss (price rises)
                    available_funds *= (1 + ret * position)
                    equity[t] = available_funds
                    trade_log.append({
                        "time": prices.index[t], "trade": "close short SL",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None

        # Step 2: Process new signals only when meta_prob > threshold
        if prob > prob_threshold:
            if signal == 1:  # Peak → SHORT
                if position > 0 and entry_px is not None:
                    # Close long before flipping short
                    ret = (cur_px - entry_px) / entry_px
                    available_funds *= (1 + ret * position)
                    trade_log.append({
                        "time": prices.index[t], "trade": "close long (flip)",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None

                # Open / add short position
                if entry_px is None:
                    entry_px = cur_px
                else:
                    entry_px = (entry_px * abs(position) + cur_px * prob) / (abs(position) + prob)
                position -= prob
                trade_log.append({
                    "time": prices.index[t], "trade": "short",
                    "trade_price": cur_px, "average_price": entry_px,
                    "trade_position": abs(prob), "current_total_position": position
                })

            elif signal == -1:  # Valley → LONG
                if position < 0 and entry_px is not None:
                    # Close short before flipping long
                    ret = (cur_px - entry_px) / entry_px
                    available_funds *= (1 + ret * position)  # position < 0
                    trade_log.append({
                        "time": prices.index[t], "trade": "close short (flip)",
                        "trade_price": cur_px, "average_price": entry_px,
                        "trade_position": abs(position), "current_total_position": 0
                    })
                    position, entry_px = 0.0, None

                # Open / add long position
                if entry_px is None:
                    entry_px = cur_px
                else:
                    entry_px = (entry_px * abs(position) + cur_px * prob) / (abs(position) + prob)
                position += prob
                trade_log.append({
                    "time": prices.index[t], "trade": "long",
                    "trade_price": cur_px, "average_price": entry_px,
                    "trade_position": abs(prob), "current_total_position": position
                })
                
        # Step 3: Mark-to-market current equity
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
    return returns_series, trade_df


def equity_curve_comparison(equity_curve_1,equity_curve_2,name_1,name_2):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_1, label=name_1, linewidth=2)
    plt.plot(equity_curve_2, label=name_2, linewidth=2)
    plt.title("Equity Curve Comparison")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


#example of simple model and backtest with trailing stop loss

df = together.copy()
train_df, test_df = embargo_split(df.copy(), 0.1, 30)
testset,reducing_feature_model,sample_X = run_rf_pipeline_from_split(
    train_df = train_df,
    test_df = test_df,
    shap_sample_size=200,
    alpha=0.4
)

explainer = shap.TreeExplainer(reducing_feature_model)
shap_values = explainer.shap_values(sample_X)

shap_values_class1 = shap_values[:, :, 1]

shap.summary_plot(shap_values_class1, sample_X, plot_type="bar")

shap.summary_plot(shap_values_class1, sample_X)


test_equity_curve, trade_log = strategy_backtesting_trailing_sl(
    testset["signals"], 
    testset["price"], 
    higher_frequency_data, 
    testset["low"],
    testset["high"], 
    testset["meta_label_probs"],
    atr = testset["ATR"],
    sl_k=3,
    tp_k=3)
strategy_equity_curve = test_equity_curve.copy()
actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
equity_curve_comparison(strategy_equity_curve,cumulative_equity_curve,"strategy","actual")


#example of simple model and backtest with simple average change rate to sl and tp
df = together.copy()
train_df, test_df = embargo_split(df.copy(), 0.1, 30)
testset,reducing_feature_model,sample_X = run_rf_pipeline_from_split(
    train_df = train_df,
    test_df = test_df,
    shap_sample_size=200,
    alpha=0.5
)
test_equity_curve, trade_log = strategy_backtesting_average_change_rate(
    predictions=testset["signals"],
    prices=testset["price"],                 
    meta_probs=testset["meta_label_probs"],               # array-like, same length as prices
    train_set_price_event=train_df,    # DataFrame with columns ['event','price']
    prob_threshold=0.3)
strategy_equity_curve = test_equity_curve.copy()
actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
equity_curve_comparison(strategy_equity_curve,cumulative_equity_curve,"strategy","actual")


#example of HMM model and backtest with trailing stop loss
df = together.copy()
df['returns']  = df['price'].pct_change()
df['log_ret2'] = np.log(df['returns']**2 + 1e-8)
df.dropna(subset=['returns'], inplace=True)
art = run_hmm_rf_pipeline_from_split(
    train_df = train_df,
    test_df = test_df,
    embargo_period=30,
    top_k_shap=20,
    shap_sample_size=200,
    alpha=0.5,
    n_components=3
)

testset = art['testset']
test_equity_curve, trade_log = strategy_backtesting_trailing_sl(
    testset["signals"], 
    testset["price"], 
    higher_frequency_data, 
    testset["low"],
    testset["high"], 
    testset["meta_label_probs"],
    atr = testset["ATR"],
    sl_k=3,
    tp_k=3)

strategy_equity_curve = test_equity_curve.copy()
actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
cumulative_equity_curve = (1 + actual_equity_curve).cumprod()

equity_curve_comparison(strategy_equity_curve,cumulative_equity_curve,"strategy","actual")           


#example of HMM model and backtest with simple average change rate to sl and tp
df = together.copy()
df['returns']  = df['price'].pct_change()
df['log_ret2'] = np.log(df['returns']**2 + 1e-8)
df.dropna(subset=['returns'], inplace=True)
train_df, test_df = embargo_split(df.copy(), 0.1, 30)
art = run_hmm_rf_pipeline_from_split(
    train_df = train_df,
    test_df = test_df,
    embargo_period=30,
    top_k_shap=20,
    shap_sample_size=200,
    alpha=0.5,
    n_components=2
)

testset = art['testset']

test_equity_curve, trade_log = strategy_backtesting_average_change_rate(
    predictions=testset["signals"],
    prices=testset["price"],                   # pd.Series (DatetimeIndex)
    meta_probs=testset["meta_label_probs"],               # array-like, same length as prices
    train_set_price_event=train_df,    # DataFrame with columns ['event','price']
    prob_threshold=0.3)
strategy_equity_curve = test_equity_curve.copy()
actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
equity_curve_comparison(strategy_equity_curve,cumulative_equity_curve,"strategy","actual")


#%%

testset = art['testset']

test_equity_curve, trade_log = strategy_backtesting_average_change_rate(
    predictions=testset["signals"],
    prices=testset["price"],                   # pd.Series (DatetimeIndex)
    meta_probs=testset["meta_label_probs"],               # array-like, same length as prices
    train_set_price_event=train_df,    # DataFrame with columns ['event','price']
    prob_threshold=0.3)
strategy_equity_curve = test_equity_curve.copy()
actual_equity_curve = testset["price"].pct_change().fillna(0).copy()
cumulative_equity_curve = (1 + actual_equity_curve).cumprod()
equity_curve_comparison(strategy_equity_curve,cumulative_equity_curve,"strategy","actual")



# 年化因子：hourly（按 6.5 小时 * 252 天）
PERIODS_PER_YEAR = int(252 * 6.5)   # ≈ 1638

def performance_metrics(equity_curve: pd.Series,
                        periods_per_year: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    """
    equity_curve: 资金曲线（pd.Series），index = 时间，values = 组合价值
    """
    # 1) 每小时收益
    returns = equity_curve.pct_change().dropna()

    # 2) Sharpe
    mean_ret = returns.mean()
    vol = returns.std(ddof=1)
    sharpe = (mean_ret / vol * np.sqrt(periods_per_year)) if vol != 0 else np.nan

    # 3) Sortino（只看负收益）
    downside = returns[returns < 0]
    downside_vol = downside.std(ddof=1)
    sortino = (mean_ret / downside_vol * np.sqrt(periods_per_year)) if downside_vol != 0 else np.nan

    # 4) Max Drawdown
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    # 5) CAGR
    total_periods = len(equity_curve) - 1
    if total_periods > 0:
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        years = total_periods / periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    else:
        total_return, cagr = np.nan, np.nan

    # 6) Calmar = CAGR / |MaxDD|
    if max_dd is not None and max_dd != 0 and not np.isnan(max_dd) and not np.isnan(cagr):
        calmar = cagr / abs(max_dd)
    else:
        calmar = np.nan

    # 7) 汇总成 DataFrame，保留两到四位小数
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
    # 美化一下显示
    df["Value_rounded"] = df["Value"].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    return df

# 使用示例
metrics_df = performance_metrics(strategy_equity_curve)
print(metrics_df)

plt.figure(figsize=(10, 6))

# 按 regime 分组并绘制
for regime, group in testset.groupby("regime"):
    plt.hist(
        group["returns"], 
        bins=30, 
        alpha=0.5,           # 透明度让多图重叠更清楚
        label=f"Regime {regime}"
    )
plt.ylim(0, 50)

plt.title("Histogram of Returns by Regime")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()






from pykalman import KalmanFilter

def kalman_filter_price(
    price,
    Q=1e-5,              # Process (state) noise variance
    R=1e-2,              # Observation noise variance
    x0=None,             # Initial state mean
    P0=1.0,              # Initial state covariance
    em=False,            # Whether to estimate Q and R using EM
    em_iters=20
):
    """
    Apply a one-dimensional Kalman filter to a price series.

    State-space model:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   (state equation)
        y_t = x_t     + v_t,   v_t ~ N(0, R)   (observation equation)

    where:
        x_t : latent (hidden) true price / trend
        y_t : observed price

    Parameters
    ----------
    price : array-like or pandas.Series
        Observed price series.
    Q : float
        Process (transition) noise variance.
        Larger Q allows the state to change more rapidly.
    R : float
        Observation noise variance.
        Larger R leads to stronger smoothing.
    x0 : float or None
        Initial state mean. If None, the first price is used.
    P0 : float
        Initial state covariance.
    em : bool
        If True, use the EM algorithm to estimate Q and R from data.
    em_iters : int
        Number of EM iterations.

    Returns
    -------
    xhat : pandas.Series
        Filtered state mean (smoothed price / trend).
    Phat : pandas.Series
        Filtered state variance (uncertainty).
    kf : KalmanFilter
        Fitted KalmanFilter object.
    """

    # Ensure the input is a pandas Series
    if isinstance(price, (list, tuple, np.ndarray)):
        price = pd.Series(price)
    elif not isinstance(price, pd.Series):
        raise TypeError("price must be a 1D array-like object or pandas Series")

    # Convert observations to numpy array
    y = price.astype(float).values

    # pykalman expects observations with shape (n_timesteps, n_dim_obs)
    y_obs = y.reshape(-1, 1)

    # Initialize state mean
    if x0 is None:
        x0 = float(y[0])

    # Define the Kalman filter
    kf = KalmanFilter(
        transition_matrices=np.array([[1.0]]),      # State transition matrix
        observation_matrices=np.array([[1.0]]),     # Observation matrix
        initial_state_mean=np.array([x0]),
        initial_state_covariance=np.array([[P0]]),
        transition_covariance=np.array([[Q]]),
        observation_covariance=np.array([[R]])
    )

    # Optionally estimate Q and R using EM
    if em:
        kf = kf.em(y_obs, n_iter=em_iters)

    # Run the Kalman filter (forward recursion)
    state_means, state_covariances = kf.filter(y_obs)

    # Convert results back to pandas Series
    xhat = pd.Series(
        state_means[:, 0],
        index=price.index,
        name="kalman_state_mean"
    )

    Phat = pd.Series(
        state_covariances[:, 0, 0],
        index=price.index,
        name="kalman_state_variance"
    )

    return xhat, Phat, kf

xhat, Phat, kf = kalman_filter_price(data["close"], Q=1e-5, R=1e-2)





df = pd.DataFrame({"price": data["close"], "kalman": xhat})


df_plot = df.loc["2025-05-01":]




ax = df_plot.plot(
    title="Kalman Filter",
    figsize=(14, 6),
    lw=1
)
ax.set_xlabel("Time")
ax.set_ylabel("Price")
plt.show()



def wavelet_denoise(
    signal: pd.Series,
    wavelet: str = "db6",
    threshold_scale: float = 0.1,
    mode: str = "per"
) -> pd.Series:
    """
    Denoise a time series using wavelet shrinkage.

    Parameters
    ----------
    signal : pd.Series
        Input noisy signal (e.g. price or return series).
    wavelet : str
        Wavelet type (e.g. 'db6').
    threshold_scale : float
        Scale factor for the threshold.
        Larger values -> stronger denoising.
    mode : str
        Signal extension mode ('per' = periodic).

    Returns
    -------
    pd.Series
        Denoised signal.
    """

    # Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(signal.values, wavelet, mode=mode)

    # Estimate threshold based on the maximum coefficient magnitude
    threshold = threshold_scale * np.max(np.abs(coeffs[1]))

    # Apply soft-thresholding to detail coefficients
    coeffs_denoised = [coeffs[0]]  # keep approximation coefficients
    for c in coeffs[1:]:
        c_thresh = pywt.threshold(c, threshold, mode="soft")
        coeffs_denoised.append(c_thresh)

    # Inverse DWT to reconstruct the signal
    reconstructed = pywt.waverec(coeffs_denoised, wavelet, mode=mode)

    # Match original length
    reconstructed = reconstructed[:len(signal)]

    return pd.Series(reconstructed, index=signal.index, name="wavelet_denoised")



wavelet = "db6"
threshold_scales = [0.1, 0.04]

fig, axes = plt.subplots(len(threshold_scales), 1, figsize=(14, 6), sharex=True)

N = 1000 

for i, scale in enumerate(threshold_scales):

    denoised_signal = wavelet_denoise(
        signal=data["close"],
        wavelet=wavelet,
        threshold_scale=scale
    )

    # Slice only the last N observations for plotting
    close_plot = data["close"].iloc[-N:]
    denoised_plot = denoised_signal.iloc[-N:]

    close_plot.plot(
        color="blue",
        alpha=0.6,
        lw=2,
        label="Original Signal",
        ax=axes[i]
    )

    denoised_plot.plot(
        color="black",
        lw=1,
        label="DWT Smoothing",
        ax=axes[i]
    )

    axes[i].set_title(f"Wavelet Denoising (db6), Threshold Scale = {scale:.2f}")
    axes[i].legend()

plt.tight_layout()
plt.show()


import matplotlib.colors as mcolors
from mlfinpy.labeling import trend_scanning_labels

aapl_close = data["close"][-2000:]

t_events = aapl_close.index

tr_scan_labels = trend_scanning_labels(aapl_close, t_events, look_forward_window=80, min_sample_length=3)



price = aapl_close.loc[tr_scan_labels.index]
tvals = tr_scan_labels["t_value"]
norm = mcolors.TwoSlopeNorm(
    vmin=np.percentile(tvals, 5),
    vcenter=0.0,
    vmax=np.percentile(tvals, 95)
)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    price.index,
    price.values,
    color="lightgray",
    linewidth=1,
    zorder=1,
    label="Price"
)
sc = ax.scatter(
    tr_scan_labels.index,
    price.values,
    c=tvals,
    cmap="viridis",    
    norm=norm,
    s=35,
    zorder=2
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Trend t-statistic")

ax.set_title("Trend Scanning Label Visualization")
ax.set_xlabel("Time")
ax.set_ylabel("Price")

plt.tight_layout()
plt.show()



tau = 1.5

tr_scan_labels["trend_label"] = np.where(
    tr_scan_labels["t_value"] >= tau,  1,
    np.where(
        tr_scan_labels["t_value"] <= -tau, -1, 0
    )
)


price = aapl_close.loc[tr_scan_labels.index]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(price.index, price, color="lightgray", lw=1)

# no-trend
mask_nt = tr_scan_labels["trend_label"] == 0
ax.scatter(
    tr_scan_labels.index[mask_nt],
    price.loc[mask_nt],
    color="gray",
    alpha=0.5,
    s=20,
    label="No trend"
)

mask_tr = tr_scan_labels["trend_label"] != 0
norm = mcolors.TwoSlopeNorm(
    vmin=-np.max(abs(tr_scan_labels["t_value"])),
    vcenter=0,
    vmax=np.max(abs(tr_scan_labels["t_value"]))
)

sc = ax.scatter(
    tr_scan_labels.index[mask_tr],
    price.loc[mask_tr],
    c=tr_scan_labels.loc[mask_tr, "t_value"],
    cmap="viridis",
    norm=norm,
    s=40,
    label="Trend"
)

plt.colorbar(sc, ax=ax, label="Trend t-statistic")
ax.legend()
ax.set_title(f"Trend Scanning with t-value Threshold = {tau}")
plt.tight_layout()
plt.show()









