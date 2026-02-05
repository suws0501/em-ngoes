import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from ib_insync import IB, Stock, util
import asyncio
import time
from dateutil.relativedelta import relativedelta
from ib_insync import Index
from ib_insync import Option

def fetch_ibkr_data_simple(
    symbol: str = "TSLA",
    duration: str = "3 Y",
    bar_size: str = "1 hour",
    use_rth: bool = False,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 3
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Interactive Brokers (IBKR) and
    disconnect immediately after fetching.

    This simplified version:
    - Connects to TWS or IB Gateway
    - Qualifies the contract
    - Requests historical bars (OHLCV)
    - Returns a pandas DataFrame
    - Disconnects once data is retrieved

    Parameters
    ----------
    symbol : str, default="TSLA"
        Stock ticker symbol (e.g., 'TSLA', 'AAPL').
    duration : str, default="3 Y"
        Lookback window (e.g., '1 D', '6 M', '3 Y').
    bar_size : str, default="1 hour"
        Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day').
    use_rth : bool, default=False
        Whether to restrict data to Regular Trading Hours.
    host : str, default="127.0.0.1"
        Host address for TWS or IB Gateway.
    port : int, default=7497
        Port number for TWS/Gateway (7497 = paper trading).
    client_id : int, default=4
        Unique ID for API client session.

    Returns
    -------
    pd.DataFrame
        Historical OHLCV data with datetime index and columns:
        ['open', 'high', 'low', 'close', 'volume'].
    """
    # Create IB connection object
    ib = IB()
    
    try:
        # If a loop is already running, this will raise; we then startLoop()
        if asyncio.get_event_loop().is_running():
            util.startLoop()
    except RuntimeError:
        # no current event loop; fine to proceed
        pass
    
    ib.connect(host, port, clientId=client_id)

    # Define contract (US stock on SMART routing)
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Pause briefly to avoid request cancellation
    time.sleep(0.2)

    # Request historical data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",       # empty = "up to now"
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=2,
        keepUpToDate=False
    )

    # Convert to pandas DataFrame
    data = util.df(bars)
    data.set_index("date", inplace=True)

    # Disconnect immediately after fetching
    ib.disconnect()

    # Return only OHLCV columns
    return data[["open", "high", "low", "close", "volume"]]

def fetch_ibkr_data_chunked(
    symbol: str = "TSLA",
    total_duration: str = "3 Y",      
    chunk_duration: str = "6 M",      
    bar_size: str = "5 mins",
    what_to_show: str = "TRADES",
    use_rth: bool = False,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 7,
    sleep_sec: float = 0.5
) -> pd.DataFrame:
    """
    Fetch high-frequency historical OHLCV over a long window by chunking requests.

    Strategy
    --------
    - Connect once
    - Pull the most-recent chunk first (endDateTime="")
    - Then move the endDateTime backward to the earliest timestamp of the last chunk - 1 second
    - Repeat until the requested total_duration is covered
    - Concatenate, drop duplicates, sort by time

    Parameters
    ----------
    symbol : str
        Ticker, e.g., 'TSLA'.
    total_duration : str
        Total lookback horizon to cover (e.g., '3 Y').
    chunk_duration : str
        Duration per historical request (e.g., '6 M', '1 M').
        Pick based on IB limits for the given bar_size.
    bar_size : str
        '1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day', etc.
    what_to_show : str
        'TRADES', 'MIDPOINT', 'BID', 'ASK', ...
    use_rth : bool
        Restrict to Regular Trading Hours.
    host, port, client_id : connection params
    sleep_sec : float
        Sleep between requests to avoid pacing/cancellation.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed OHLCV DataFrame covering (approx.) total_duration
        at the requested bar_size.
    """
    def _fmt_ib_end(dt: pd.Timestamp) -> str:
        """
        Convert a pandas Timestamp to IB endDateTime string "YYYYMMDD HH:MM:SS".
        """
        if pd.isna(dt):
            return ""

        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_convert(None)
        return dt.strftime("%Y%m%d %H:%M:%S")

    def parse_duration(duration: str) -> relativedelta:
        """
        Convert IB-style duration string (e.g. '3 Y', '6 M', '10 D')
        into a relativedelta object.
        """
        num, unit = duration.split()
        num = int(num)
        unit = unit.upper()
        if unit.startswith("Y"):
            return relativedelta(years=num)
        elif unit.startswith("M"):
            return relativedelta(months=num)
        elif unit.startswith("W"):
            return relativedelta(weeks=num)
        elif unit.startswith("D"):
            return relativedelta(days=num)
        else:
            raise ValueError(f"Unsupported duration unit: {unit}")
            
    ib = IB()
    
    
    try:
        # If a loop is already running, this will raise; we then startLoop()
        if asyncio.get_event_loop().is_running():
            util.startLoop()
    except RuntimeError:
        # no current event loop; fine to proceed
        pass
    
    ib.connect(host, port, clientId=client_id)
    try:
        ib.reqMarketDataType(3)

        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        time.sleep(0.2)

        frames = []
        end_dt = ""  # IB will interpret as 'now'
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=chunk_duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
            keepUpToDate=False
        )
        df = util.df(bars)
        if df.empty:
            raise RuntimeError("Empty data on first chunk; check permissions/params.")
        df.set_index("date", inplace=True)
        frames.append(df[["open", "high", "low", "close", "volume"]])

        def _span_ok(frames_list, total_duration: str) -> bool:
            """
            Check if concatenated frames already cover the target total_duration.
            """
            cat = pd.concat(frames_list).sort_index()
            earliest, latest = cat.index.min(), cat.index.max()
            target_earliest = latest - parse_duration(total_duration)
            return earliest <= target_earliest

        earliest = df.index.min()

        for _ in range(200):
            if _span_ok(frames, total_duration):
                break


            next_end = _fmt_ib_end(earliest - pd.Timedelta(seconds=1))

            time.sleep(sleep_sec)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=next_end,
                durationStr=chunk_duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=2,
                keepUpToDate=False
            )
            df_chunk = util.df(bars)
            if df_chunk.empty:
                time.sleep(sleep_sec * 2)
                continue

            df_chunk.set_index("date", inplace=True)
            frames.append(df_chunk[["open", "high", "low", "close", "volume"]])

            earliest = min(earliest, df_chunk.index.min())

        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")]
        out.sort_index(inplace=True)

        return out

    finally:
        ib.disconnect()
        
        
data = fetch_ibkr_data_simple(symbol = "AAPL",duration = "3 Y",client_id=7)
AAPL = data
path = r"data_fetch.parquet"
AAPL.to_parquet(path)



higher_frequency_data = fetch_ibkr_data_chunked(
    symbol="AAPL",
    total_duration="1 Y",
    chunk_duration="2 M",
    bar_size="5 mins",
    use_rth=False
)
