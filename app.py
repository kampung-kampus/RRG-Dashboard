import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy import interpolate

# Set up page configurations
st.set_page_config(layout="wide", page_title="US Sector RRG Dashboard")
st.title("📊 US Sector Relative Rotation Graph (RRG)")
st.write("Track the relative strength and momentum of US sectors against the S&P 500.")

# Configuration inputs in sidebar
st.sidebar.header("RRG Settings")
period = st.sidebar.selectbox("Historical Period", ['1y', '6mo', '2y'], index=0)
window = st.sidebar.slider("Rolling Calculation Window", min_value=5, max_value=30, value=14)
tail = st.sidebar.slider("Tail Length (Weeks)", min_value=1, max_value=15, value=5)

# Hardcoded US Sector ETFs and S&P 500 Benchmark
tickers = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
benchmark = '^GSPC'

# Helper function to compute smooth tail curves using spline interpolation
def get_line_points(x, y):
    if len(x) < 3:  
        return x, y
    try:
        tck, _ = interpolate.splprep([x, y], s=0)
        t = np.linspace(0, 1, 50)
        line_x, line_y = interpolate.splev(t, tck)
        return line_x, line_y
    except Exception:
        return x, y  

def get_status(x, y):
    if x < 100 and y < 100: return 'Lagging'
    elif x >= 100 and y >= 100: return 'Leading'
    elif x < 100 and y >= 100: return 'Improving'
    else: return 'Weakening'

# Load stock market data safely handling yfinance multi-index structures
@st.cache_data(ttl=3600)  
def load_data(tickers_list, bench, prd):
    all_tickers = tickers_list + [bench]
    # Use group_by="ticker" to cleanly separate columns per asset
    raw_data = yf.download(all_tickers, period=prd, interval="1wk", group_by="ticker")
    
    # Extract Adjusted Close for each individual asset safely
    t_data = pd.DataFrame()
    for t in tickers_list:
        if t in raw_data.columns.levels[0]:
            t_data[t] = raw_data[t]['Adj Close']
            
    b_data = raw_data[bench]['Adj Close']
    return t_data, b_data

try:
    tickers_data, benchmark_data = load_data(tickers, benchmark, period)
    
    rsr_dict = {}
    rsm_dict = {}
    common_idx = None
    
    for ticker in tickers:
        # RS = 100 * (Price / Benchmark)
        rs = 100 * (tickers_data[ticker] / benchmark_data)
        
        # RS-Ratio = 100 + ((RS - Avg) / Std)
        rs_mean = rs.rolling(window=window).mean()
        rs_std = rs.rolling(window=window).std(ddof=0)
        rsr = (100 + ((rs - rs_mean) / rs_std)).dropna()
        
        # RS-Momentum = 101 + ((ROC - Avg) / Std)
        rsr_roc = 100 * ((rsr / rsr.shift(1)) - 1).dropna()
        roc_mean = rsr_roc.rolling(window=window).mean()
        roc_std = rsr_roc.rolling(window=window).std(ddof=0)
        rsm = (101 + ((rsr_roc - roc_mean) / roc_std)).dropna()
        
        # Keep track of overlapping indices
        ticker_common = rsr.index.intersection(rsm.index)
        if common_idx is None:
            common_idx = ticker_common
        else:
            common_idx = common_idx.intersection(ticker_common)
            
        rsr_dict[ticker] = rsr
        rsm_dict[ticker] = rsm

    # Filter out historical values based on complete datasets across all tickers
    available_dates = common_idx.sort_values()
    total_weeks = len(available_dates)
    
    if total_weeks <= tail + 2:
        st.error("The calculation rolling window is too wide for this short period dataset. Increase historical timeline period context in the sidebar.")
    else:
        # Time Slider to navigate history snapshots
        selected_val = st.slider("Select Historical Snapshot Date", 
                                 min_value=int(tail), 
                                 max_value=int(total_weeks - 1), 
                                 value=int(total_weeks - 1))
        
        snapshot_date = available_dates[selected_val]
        st.subheader(f"Snapshot Analysis Date: {snapshot_date.strftime('%Y-%m-%d')}")

        fig = go.Figure()

        # Build colored visual quadrant zones background 
        fig.add_vrect(x0=90, x1=100, y0=90, y1=100, fillcolor="red", opacity=0.06, layer="below", line_width=0)
        fig.add_vrect(x0=100, x1=110, y0=90, y1=100, fillcolor="yellow", opacity=0.06, layer="below", line_width=0)
        fig.add_vrect(x0=100, x1=110, y0=100, y1=110, fillcolor="green", opacity=0.06, layer="below", line_width=0)
        fig.add_vrect(x0=90, x1=100, y0=100, y1=110, fillcolor="blue", opacity=0.06, layer="below", line_width=0)

        # Draw grid crosshairs dividing origin at (100, 100)
        fig.add_hline(y=100, line_dash="dash", line_color="black", opacity=0.4)
        fig.add_vline(x=100, line_dash="dash", line_color="black", opacity=0.4)

        table_data = []

        # Map details for active items
        for ticker in tickers:
            t_date_range = available_dates[selected_val - tail + 1 : selected_val + 1]
            
            x_trail = rsr_dict[ticker].loc[t_date_range].values
            y_trail = rsm_dict[ticker].loc[t_date_range].values
            
            current_x = x_trail[-1]
            current_y = y_trail[-1]
            status = get_status(current_x, current_y)
            
            table_data.append({
                "Ticker": ticker,
                "RS Ratio (X)": round(current_x, 2),
                "RS Momentum (Y)": round(current_y, 2),
                "Quadrant": status
            })

            # Formulate polished smooth display lines
            lx, ly = get_line_points(x_trail, y_trail)

            # Draw trajectory path tails
            fig.add_trace(go.Scatter(
                x=lx, y=ly, mode='lines',
                line=dict(width=2), name=ticker,
                showlegend=False, hoverinfo='skip'
            ))

            # Draw major endpoint bubble nodes
            fig.add_trace(go.Scatter(
                x=[current_x], y=[current_y],
                mode='markers+text',
                marker=dict(size=11, symbol='circle'),
                text=[ticker], textposition="top center",
                name=ticker,
                hovertemplate=f"<b>{ticker}</b><br>RS Ratio: {current_x:.2f}<br>RS Momentum: {current_y:.2f}<br>Status: {status}<extra></extra>"
            ))

        # Adjust general dimensions
        fig.update_layout(
            xaxis_title="JdK RS Ratio (Trend)",
            yaxis_title="JdK RS Momentum (Velocity)",
            xaxis=dict(range=[92, 108]),
            yaxis=dict(range=[92, 108]),
            width=850, height=600,
            margin=dict(l=30, r=30, t=30, b=30),
            hovermode='closest'
        )

        # Label Quadrants
        fig.add_annotation(x=94, y=106, text="<b>IMPROVING</b>", showarrow=False, font=dict(color="blue", size=13))
        fig.add_annotation(x=106, y=106, text="<b>LEADING</b>", showarrow=False, font=dict(color="green", size=13))
        fig.add_annotation(x=106, y=94, text="<b>WEAKENING</b>", showarrow=False, font=dict(color="orange", size=13))
        fig.add_annotation(x=94, y=94, text="<b>LAGGING</b>", showarrow=False, font=dict(color="red", size=13))

        # Generate side-by-side structures
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Sector Tracking Breakdowns")
            df_status = pd.DataFrame(table_data)
            st.dataframe(df_status.set_index("Ticker"), use_container_width=True)

except Exception as e:
    st.error(f"An unexpected error occurred during execution: {e}")

