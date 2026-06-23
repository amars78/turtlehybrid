import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- pykrx 예외 처리 및 로드 ---
try:
    from pykrx import stock as krx
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

# --- 페이지 설정 ---
st.set_page_config(page_title="CAN SLIM x 터틀 실전 매니저", layout="wide")
st.title("🦅 CAN SLIM x 🐢 터틀 트레이딩 실전 자산 매니저")
st.markdown("""
티커(종목코드)를 입력하면 **국내 주식(Name)과 해외 주식(Short Name)**을 자동으로 찾아옵니다.
포지션을 입력하여 실시간 손절 및 추가매수(피라미딩) 타이밍을 관리하세요.
""")

# --- 고도화된 종목명 조회 함수 ---
@st.cache_data(ttl=86400)
def get_stock_name(ticker: str) -> str:
    """국내 종목(.KS/.KQ)은 pykrx로 한글명, 해외 종목은 yfinance로 영문명 조회"""
    ticker = ticker.strip().upper()
    if not ticker:
        return ""
        
    # 1. 국내 주식 처리 (pykrx 활용)
    if PYKRX_AVAILABLE and (ticker.endswith(".KS") or ticker.endswith(".KQ")):
        code = ticker.split(".")[0]
        try:
            name = krx.get_market_ticker_name(code)
            if name and name.strip():
                return name
        except Exception:
            pass
        return ticker

    # 2. 해외 주식 처리 (yfinance 활용)
    try:
        tk = yf.Ticker(ticker)
        # .info 가 블로킹될 때를 대비해 가벼운 fast_info나 구형 데이터 구조 교차 검증
        info = tk.info
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return ticker # 실패 시 티커 자체를 반환

# --- 기본 금융 데이터 함수들 ---
def get_benchmark_ticker(ticker: str) -> str:
    if ticker.endswith(".KS"): return "^KS11"
    elif ticker.endswith(".KQ"): return "^KQ11"
    return "^GSPC"

@st.cache_data(ttl=3600)
def load_price_history(ticker: str, days: int = 500) -> pd.DataFrame | None:
    start = datetime.today() - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start, end=datetime.today(), progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df
    except Exception: return None

@st.cache_data(ttl=3600)
def get_market_trend(benchmark_ticker: str) -> dict:
    df = load_price_history(benchmark_ticker, days=400)
    if df is None or len(df) < 200: return {"status": "확인불가", "detail": "데이터 부족"}
    close = df["Close"]
    sma50, sma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
    sma200_prev = close.rolling(200).mean().iloc[-20] if len(df) > 220 else sma200
    if (close.iloc[-1] > sma50) and (close.iloc[-1] > sma200) and (sma200 > sma200_prev):
        return {"status": "🟢 상승추세", "detail": "지수가 50/200일선 위, 200일선 상승 중"}
    elif (close.iloc[-1] < sma50) and (close.iloc[-1] < sma200) and (sma200 < sma200_prev):
        return {"status": "🔴 하락추세", "detail": "지수가 이평선 아래, 200일선 하락 중"}
    return {"status": "🟡 중립/전환구간", "detail": "이평선 신호 혼재"}

def compute_rs_raw(df: pd.DataFrame) -> float | None:
    close = df["Close"].dropna()
    if len(close) < 253: return None
    def r(e, s): return (close.iloc[e] / close.iloc[s]) - 1.0 if close.iloc[s] != 0 else 0
    return 2 * r(-1, -64) + r(-64, -127) + r(-127, -190) + r(-190, -253)

def rs_rating_from_raw(raw_scores: dict) -> dict:
    valid = {k: v for k, v in raw_scores.items() if v is not None}
    if len(valid) <= 1: return {k: 50 for k in raw_scores}
    values = sorted(valid.values())
    return {k: (max(1, min(99, round((sum(1 for x in values if x <= v) / len(values)) * 98) + 1)) if v is not None else None) for k, v in raw_scores.items()}

def compute_volume_signal(df: pd.DataFrame) -> dict:
    recent = df.tail(50).copy()
    recent["change"] = recent["Close"].diff()
    up_vol = recent.loc[recent["change"] > 0, "Volume"].sum()
    down_vol = recent.loc[recent["change"] < 0, "Volume"].sum()
    ratio = up_vol / down_vol if down_vol > 0 else 1.0
    return {"ratio": round(ratio, 2), "signal": "🟢 매집" if ratio >= 1.2 else "🔴 분산" if ratio <= 0.8 else "🟡 중립"}

@st.cache_data(ttl=3600)
def load_and_process_data(ticker, entry_w, exit_w):
    df = load_price_history(ticker, days=500)
    if df is None: return None
    try:
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['SMA_150'] = df['Close'].rolling(150).mean()
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df['52W_High'], df['52W_Low'] = df['High'].rolling(250).max(), df['Low'].rolling(250).min()
        df['SMA_200_Trend'] = df['SMA_200'] > df['SMA_200'].shift(20)
        df['Entry_High'] = df['High'].rolling(entry_w).max().shift(1)
        df['Exit_Low'] = df['Low'].rolling(exit_w).min().shift(1)
        ranges = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift(1)), np.abs(df['Low']-df['Close'].shift(1))], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(20).mean()
        return df
    except Exception: return None

# --- 사이드바 설정 ---
st.sidebar.header("⚙️ 시스템 및 자금 관리 설정")
system_type = st.sidebar.radio("터틀 시스템 선택", ("시스템 1 (20일 돌파)", "시스템 2 (55일 돌파)"))
entry_window, exit_window = (20, 10) if system_type == "시스템 1 (20일 돌파)" else (55, 20)
account_size = st.sidebar.number_input("총 투자 자본금", value=100000, step=10000)
risk_per_trade = st.sidebar.slider("1유닛 리스크 비율 (%)", 0.5, 5.0, 1.0, 0.1) / 100
max_units = st.sidebar.slider("최대 피라미딩 유닛 수", 1, 4, 4)
apply_market_filter = st.sidebar.checkbox("시장이 하락추세면 매수등급 자동 하향", value=True)

# --- [세션 상태] 실전 포지션 초기 데이터베이스 ---
if "active_positions" not in st.session_state:
    st.session_state.active_positions = pd.DataFrame([
        {"티커": "AAPL", "종목명": "Apple Inc.", "실제최초매수가": 175.0, "현재보유유닛": 2},
        {"티커": "005930.KS", "종목명": "삼성전자", "실제최초매수가": 72000.0, "현재보유유닛": 1}
    ])

# --- 탭 구성 ---
tab0, tab1, tab2 = st.tabs(["🔥 1. 실전 보유 포지션 관리", "📊 2. CAN SLIM 관심종목 스캐너", "📈 3. 개별 종목 융합 차트"])

# =========================================================
# 탭 0: 실전 보유 포지션 관리 (종목명 자동 완성 기능 탑재)
# =========================================================
with tab0:
    st.subheader("🛠️ 보유 포지션 입력 및 편집")
    st.caption("💡 티커 열에 코드를 입력하고 빈 곳을 누르면 '종목명'이 자동으로 입력됩니다. (국내 주식은 뒤에 .KS 또는 .KQ 필수)")
    
    # 데이터 에디터 배치 및 열 제어
    edited_df = st.data_editor(
        st.session_state.active_positions, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "종목명": st.column_config.TextColumn("종목명 (자동 입력)", disabled=True),
            "티커": st.column_config.TextColumn("티커 (정확히 입력)", required=True),
            "실제최초매수가": st.column_config.NumberColumn("최초 매수가", required=True, min_value=0.0),
            "현재보유유닛": st.column_config.NumberColumn("현재 유닛 수", required=True, min_value=1, max_value=4, default=1)
        }
    )
    
    # 사용자가 티커를 새로 썼을 때 종목명을 동적으로 매핑하는 백엔드 로직
    names_updated = False
    for idx, row in edited_df.iterrows():
        t = str(row["티커"]).strip().upper() if pd.notna(row["티ker"] if "티ker" in row else row.get("티커")) else ""
        if t:
            current_name = row.get("종목명")
            if pd.isna(current_name) or current_name == "" or current_name == t:
                fetched_name = get_stock_name(t)
                edited_df.at[idx, "종목명"] = fetched_name
                names_updated = True
                
    st.session_state.active_positions = edited_df
    if names_updated:
        st.rerun() # 자동 입력을 UI에 즉시 반영

    st.divider()
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")

    real_management_data = []
    for _, row in edited_df.iterrows():
        ticker = str(row["티커"]).strip().upper() if pd.notna(row["티커"]) else ""
        init_price = row["실제최초매수가"]
        held_units = row["현재보유유닛"]
        stock_name = row["종목명"]
        
        if not ticker or pd.isna(init_price) or pd.isna(held_units): continue
        
        df = load_and_process_data(ticker, entry_window, exit_window)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = float(latest['Close'])
            atr = float(latest['ATR'])
            exit_channel = float(latest['Exit_Low'])
            
            # 터틀 계산식
            latest_unit_price = init_price + (0.5 * atr * (held_units - 1))
            actual_stop_loss = latest_unit_price - (2 * atr)
            next_pyramid_price = init_price + (0.5 * atr * held_units)
            
            if c_price <= actual_stop_loss:
                action_guide = "🚨 즉시 매도 (2N 실전 손절선 탈락!)"
                status_color = "🔴"
            elif c_price <= exit_channel:
                action_guide = "🚨 즉시 매도 (채널 청산선 탈락!)"
                status_color = "🔴"
            elif held_units < max_units and c_price >= next_pyramid_price:
                action_guide = f"➕ 증액 추천 (+1유닛 추가 매수 기준가: {round(next_pyramid_price, 2)})"
                status_color = "🔵"
            else:
                action_guide = "🟢 정상 보유 (추세 유지 중)"
                status_color = "🟢"
                
            pnl_pct = ((c_price - init_price) / init_price) * 100
            
            real_management_data.append({
                "상태": status_color,
                "종목명": stock_name if stock_name else get_stock_name(ticker),
                "티커": ticker,
                "현재가": round(c_price, 2),
                "최초 매수가": round(init_price, 2),
                "보유 유닛": f"{held_units} / {max_units}",
                "수익률": f"{pnl_pct:+.2f}%",
                "실전 손절가(2N)": round(actual_stop_loss, 2),
                "채널 청산선": round(exit_channel, 2),
                "다음 증액 목표가": round(next_pyramid_price, 2) if held_units < max_units else "최대 유닛",
                "실시간 대응 가이드": action_guide
            })
            
    if real_management_data:
        st.dataframe(pd.DataFrame(real_management_data), use_container_width=True, hide_index=True)
    else:
        st.info("포지션을 입력하시면 실시간 대응 분석표가 이곳에 출력됩니다.")

# =========================================================
# 탭 1 & 2: 관심종목 스캐너 및 차트 (종목명 노출 강화)
# =========================================================
with tab1:
    st.subheader("🌐 시장 방향성(M) 및 관심종목 발굴")
    tickers_input = st.text_input("스캔할 관심 종목 리스트", "AAPL, MSFT, NVDA, TSLA, 005930.KS")
    scan_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    scan_data = []
    for ticker in scan_tickers:
        df = load_and_process_data(ticker, entry_window, exit_window)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            scan_data.append({
                "종목명": get_stock_name(ticker),
                "티커": ticker,
                "현재가": round(latest['Close'], 2),
                "터틀 진입선": round(latest['Entry_High'], 2),
                "터틀 청산선": round(latest['Exit_Low'], 2)
            })
    if scan_data:
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True, hide_index=True)

with tab2:
    all_known_tickers = list(set(scan_tickers + [str(r.get("티커")).upper() for _, r in edited_df.iterrows() if pd.notna(r.get("티커"))]))
    selected_ticker = st.selectbox("정밀 분석 차트 대상 선택",高度 = all_known_tickers, format_func=lambda x: f"{get_stock_name(x)} ({x})")
    st.caption(f"선택된 종목: **{get_stock_name(selected_ticker)}**")
