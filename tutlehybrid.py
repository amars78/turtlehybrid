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
티커(종목코드)를 입력하면 **국내 주식 한글명과 해외 주식 이름**을 자동으로 찾아옵니다.
실전 포지션을 기록하여 실시간 손절 및 추가매수(피라미딩) 타이밍을 효과적으로 관리하세요.
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
        info = tk.info
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return ticker

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

# --- [세션 상태] 실전 포지션 초기화 ---
if "active_positions" not in st.session_state:
    st.session_state.active_positions = pd.DataFrame([
        {"티커": "AAPL", "종목명": "Apple Inc.", "실제최초매수가": 175.0, "현재보유유닛": 2},
        {"티커": "005930.KS", "종목명": "삼성전자", "실제최초매수가": 72000.0, "현재보유유닛": 1}
    ])

# --- 탭 구성 ---
tab0, tab1, tab2 = st.tabs(["🔥 1. 실전 보유 포지션 관리", "📊 2. CAN SLIM 관심종목 스캐너", "📈 3. 개별 종목 융합 차트"])

# =========================================================
# 탭 0: 실전 보유 포지션 관리
# =========================================================
with tab0:
    st.subheader("🛠️ 보유 포지션 입력 및 편집")
    st.caption("💡 티커 열에 코드를 입력하고 빈 곳을 누르면 '종목명'이 자동으로 입력됩니다.")
    
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
    
    # 무한 루프 없는 안전한 자동완성 백엔드 로직
    names_updated = False
    for idx, row in edited_df.iterrows():
        t = str(row.get("티커", "")).strip().upper()
        if t:
            current_name = row.get("종목명", "")
            if pd.isna(current_name) or current_name == "":
                fetched_name = get_stock_name(t)
                edited_df.at[idx, "종목명"] = fetched_name
                names_updated = True
                
    st.session_state.active_positions = edited_df
    if names_updated:
        st.rerun()

    st.divider()
    st.subheader("🚨 실시간 보유 포지션 대응 알림판")

    real_management_data = []
    for _, row in edited_df.iterrows():
        ticker = str(row.get("티커", "")).strip().upper()
        init_price = row.get("실제최초매수가")
        held_units = row.get("currently_units" if "currently_units" in row else row.get("현재보유유닛"))
        stock_name = row.get("종목명", "")
        
        if not ticker or pd.isna(init_price) or pd.isna(held_units): continue
        
        df = load_and_process_data(ticker, entry_window, exit_window)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = float(latest['Close'])
            atr = float(latest['ATR'])
            exit_channel = float(latest['Exit_Low'])
            
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
                action_guide = f"➕ 증액 추천 (+1유닛 추가 매수 완료 기준가: {round(next_pyramid_price, 2)})"
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
# 탭 1: CAN SLIM 관심종목 스캐너
# =========================================================
with tab1:
    st.subheader("🔍 관심종목 발굴 스캐너")
    tickers_input = st.text_input("스캔할 관심 종목 리스트 (쉼표 구분)", "AAPL, MSFT, NVDA, TSLA, 005930.KS")
    scan_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    unique_benchmarks = set(get_benchmark_ticker(t) for t in scan_tickers)
    market_trend_map = {b: get_market_trend(b) for b in unique_benchmarks}
    
    raw_scores = {}
    processed_dfs = {}
    for t in scan_tickers:
        df_t = load_and_process_data(t, entry_window, exit_window)
        processed_dfs[t] = df_t
        if df_t is not None:
            raw_scores[t] = compute_rs_raw(df_t)
    rs_ratings = rs_rating_from_raw(raw_scores)

    scan_data = []
    for ticker in scan_tickers:
        df = processed_dfs.get(ticker)
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            c_price = latest['Close']
            
            cond1 = (c_price > latest['SMA_150']) and (c_price > latest['SMA_200'])
            cond2 = latest['SMA_150'] > latest['SMA_200']
            cond3 = bool(latest['SMA_200_Trend'])
            cond4 = (latest['SMA_50'] > latest['SMA_150']) and (latest['SMA_50'] > latest['SMA_200'])
            cond5 = c_price > latest['SMA_50']
            cond6 = c_price >= (latest['52W_High'] * 0.75)
            cond7 = c_price >= (latest['52W_Low'] * 1.30)
            trend_score = sum([cond1, cond2, cond3, cond4, cond5, cond6, cond7])
            
            rs_val = rs_ratings.get(ticker, 50)
            cond_rs = rs_val >= 70 if rs_val else False
            vol_info = compute_volume_signal(df)
            cond_vol = vol_info["signal"] == "🟢 매집"
            
            canslim_score = trend_score + int(cond_rs) + int(cond_vol)
            
            bench = get_benchmark_ticker(ticker)
            market_bearish = "하락추세" in market_trend_map.get(bench, {}).get("status", "")
            
            if canslim_score >= 7: base_status, base_emoji = "강력 매수 고려", "🔥"
            elif canslim_score >= 5: base_status, base_emoji = "관망/대기", "🟡"
            else: base_status, base_emoji = "추세 약함", "❄️"
            
            status_text = f"🟠 시장약세로 보류 ({canslim_score}/9)" if apply_market_filter and market_bearish and base_status == "강력 매수 고려" else f"{base_emoji} {base_status} ({canslim_score}/9)"
            
            scan_data.append({
                "종목명": get_stock_name(ticker),
                "티커": ticker,
                "CAN SLIM 점수": status_text,
                "RS Rating": rs_val if rs_val else "-",
                "수급 신호": vol_info["signal"],
                "현재가": round(c_price, 2),
                "터틀 진입선": round(latest['Entry_High'], 2),
                "터틀 청산선": round(latest['Exit_Low'], 2)
            })
    if scan_data:
        st.dataframe(pd.DataFrame(scan_data), use_container_width=True, hide_index=True)

# =========================================================
# 탭 2: 개별 종목 융합 차트 및 세부 계획
# =========================================================
with tab2:
    st.subheader("📈 개별 종목 정밀 융합 차트")
    all_known_tickers = list(set(scan_tickers + [str(r.get("티커", "")).upper() for _, r in edited_df.iterrows() if r.get("티커")]))
    all_known_tickers = [t for t in all_known_tickers if t]
    
    if all_known_tickers:
        selected_ticker = st.selectbox("분석할 종목을 선택하세요", options=all_known_tickers, format_func=lambda x: f"{get_stock_name(x)} ({x})")
        
        df_chart = load_and_process_data(selected_ticker, entry_window, exit_window)
        if df_chart is not None and not df_chart.empty:
            df_plot = df_chart.tail(200).copy()
            df_plot['Buy_Signal'] = (df_plot['Close'] > df_plot['Entry_High']) & (df_plot['Close'].shift(1) <= df_plot['Entry_High'].shift(1))
            df_plot['Sell_Signal'] = (df_plot['Close'] < df_plot['Exit_Low']) & (df_plot['Close'].shift(1) >= df_plot['Exit_Low'].shift(1))
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='가격'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_50'], line=dict(color='orange', width=1.5), name='50일선'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='purple', width=2), name='200일선'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Entry_High'], line=dict(color='blue', dash='dot'), name='터틀 진입선'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Exit_Low'], line=dict(color='red', dash='dot'), name='터틀 청산선'))
            
            buy_signals = df_plot[df_plot['Buy_Signal']]
            sell_signals = df_plot[df_plot['Sell_Signal']]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=13), name='터틀 돌파'))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=13), name='터틀 이탈'))
            
            fig.update_layout(title=f'{get_stock_name(selected_ticker)} ({selected_ticker}) — 분석 차트', xaxis_rangeslider_visible=False, template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 피라미딩 구조 테이블 복구
            st.subheader("🐢 터틀 피라미딩 & 손절 세부 계획")
            latest = df_chart.iloc[-1]
            atr = latest['ATR']
            entry_price = latest['Entry_High']
            
            if not pd.isna(atr) and not pd.isna(entry_price) and atr > 0:
                risk_amount = account_size * risk_per_trade
                unit_shares = int(risk_amount / atr)
                
                plan_rows = []
                for unit_n in range(1, max_units + 1):
                    add_price = entry_price + 0.5 * atr * (unit_n - 1)
                    plan_rows.append({
                        "유닛 단계": f"{unit_n}유닛",
                        "추가 매수 기준가": round(add_price, 2),
                        "유닛당 매수량": f"{unit_shares} 주",
                        "최종 누적 물량": f"{unit_shares * unit_n} 주",
                    })
                st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
        else:
            st.warning("해당 종목의 차트 데이터를 불러올 수 없습니다.")
    else:
        st.info("스캐너 혹은 포지션 관리 테이블에 종목을 입력하시면 차트 탭이 활성화됩니다.")
