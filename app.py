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
period = st.sidebar.selectbox("Historical Period", ['1y', '6mo', '3mo', '2y'], index=0)
window = st.sidebar.slider("Rolling Calculation Window", min_value=5, max_value=30, value=14)
tail = st.sidebar.slider("Tail Length (Weeks)", min_value=1, max_value=15, value=5)

# Hardcoded US Sector ETFs and S&P 500 Benchmark
tickers = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
benchmark = '^GSPC'

# Helper function to compute smooth tail curves using spline interpolation
def get_line_points(x, y):
    if len(x) < 3:  # Not enough points for spline interpolation
        return x, y
    try:
        tck, _ = interpolate.splprep([x, y], s=0)
        t = np.linspace(0, 1, 50)
        line_x, line_y = interpolate.splev(t, tck)
        return line_x, line_y
    except Exception:
        return x, y  # Fallback to straight lines if spline fails

def get_status(x, y):
    if x < 100 and y < 100: return 'Lagging'
    elif x >= 100 and y >= 100: return 'Leading'
    elif x < 100 and y >= 100: return 'Improving'
    else: return 'Weakening'

# Load stock market data
@st.cache_data(ttl=3600)  # Cache data for 1 hour to avoid throttling yfinance
def load_data(tickers_list, bench, prd):
    all_tickers = tickers_list + [bench]
    data = yf.download(all_tickers, period=prd, interval="1wk")['Adj Close']
    return data[tickers_list], data[bench]

try:
    tickers_data, benchmark_data = load_data(tickers, benchmark, period)
    
    # Calculate RRG Indices
    rsr_list = []
    rsm_list = []
    
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
        
        # Synchronize indices after dropping NaN elements
        common_idx = rsr.index.intersection(rsm.index)
        rsr_list.append(rsr.loc[common_idx])
        rsm_list.append(rsm.loc[common_idx])

    # Date range selection mapping from calculations
    available_dates = rsr_list[0].index
    total_weeks = len(available_dates)
    
    if total_weeks <= tail + 1:
        st.error("The calculation window is too wide for the chosen period. Try increasing the period length or reducing the window size.")
    else:
        # Time Slider to travel back in history
        selected_idx = st.slider("Select Historical Snapshot Date", 
                                 min_value=tail, 
                                 max_value=total_weeks - 1, 
                                 value=total_weeks - 1,
                                 format="YYYY-MM-DD")
        
        snapshot_date = available_dates[selected_idx]
        st.subheader(f"Snapshot Analysis Date: {snapshot_date.strftime('%Y-%m-%d')}")

        # Setup interactive Plotly RRG Chart
        fig = go.Figure()

        # Add visual quadrants background colors using shapes
        fig.add_vrect(x0=90, x1=100, y0=90, y1=100, fillcolor="red", opacity=0.08, layer="below", line_width=0)
        fig.add_vrect(x0=100, x1=110, y0=90, y1=100, fillcolor="yellow", opacity=0.08, layer="below", line_width=0)
        fig.add_vrect(x0=100, x1=110, y0=100, y1=110, fillcolor="green", opacity=0.08, layer="below", line_width=0)
        fig.add_vrect(x0=90, x1=100, y0=100, y1=110, fillcolor="blue", opacity=0.08, layer="below", line_width=0)

        # Plot quadrants axis lines at (100, 100) origin
        fig.add_hline(y=100, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_vline(x=100, line_dash="dash", line_color="black", opacity=0.5)

        table_data = []

        # Populate chart elements for each ticker
        for i, ticker in enumerate(tickers):
            # Isolate data history trailing backward from selected time index
            x_trail = rsr_list[i].iloc[selected_idx - tail + 1 : selected_idx + 1].values
            y_trail = rsm_list[i].iloc[selected_idx - tail + 1 : selected_idx + 1].values
            
            current_x = x_trail[-1]
            current_y = y_trail[-1]
            status = get_status(current_x, current_y)
            
            table_data.append({
                "Ticker": ticker,
                "RS Ratio (X)": round(current_x, 2),
                "RS Momentum (Y)": round(current_y, 2),
                "Quadrant": status
            })

            # Smooth historical tail curve calculations
            lx, ly = get_line_points(x_trail, y_trail)

            # Draw the trailing line
            fig.add_trace(go.Scatter(
                x=lx, y=ly,
                mode='lines',
                line=dict(width=2),
                name=ticker,
                showlegend=False,
                hoverinfo='skip'
            ))

            # Draw the leading node arrow/head marker
            fig.add_trace(go.Scatter(
                x=[current_x], y=[current_y],
                mode='markers+text',
                marker=dict(size=12, symbol='circle'),
                text=[ticker],
                textposition="top center",
                name=ticker,
                hovertemplate=f"<b>{ticker}</b><br>RS Ratio: {current_x:.2f}<br>RS Momentum: {current_y:.2f}<br>Status: {status}<extra></extra>"
            ))

        # Adjust layout parameters
        fig.update_layout(
            xaxis_title="JdK RS Ratio (Trend)",
            yaxis_title="JdK RS Momentum (Velocity)",
            xaxis=dict(range=[94, 106]),
            yaxis=dict(range=[94, 106]),
            width=900,
            height=650,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='closest'
        )

        # Draw annotations defining quadrant classifications
        fig.add_annotation(x=95, y=105, text="<b>IMPROVING</b> (Blue)", showarrow=False, font=dict(color="blue", size=14))
        fig.add_annotation(x=105, y=105, text="<b>LEADING</b> (Green)", showarrow=False, font=dict(color="green", size=14))
        fig.add_annotation(x=105, y=95, text="<b>WEAKENING</b> (Yellow)", showarrow=False, font=dict(color="orange", size=14))
        fig.add_annotation(x=95, y=95, text="<b>LAGGING</b> (Red)", showarrow=False, font=dict(color="red", size=14))

        # Render layout column components
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Current Sector Quadrant Allocation")
            df_status = pd.DataFrame(table_data)
            st.dataframe(df_status.set_index("Ticker"), use_container_width=True)

except Exception as e:
    st.error(f"An unexpected error occurred during execution: {e}")
