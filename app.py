import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VN Stock Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def center_table(df, color_func=None):
    """
    Trả về pandas Styler căn giữa tất cả ô và header.
    Nếu muốn highlight màu ô, truyền hàm color_func.
    """
    if color_func:
        styled = df.style.applymap(color_func)
    else:
        styled = df.style
    
    # Căn giữa nội dung
    styled = styled.set_properties(**{'text-align': 'center'})
    
    # Căn giữa header
    styled = styled.set_table_styles([{
        'selector': 'th',
        'props': [('text-align', 'center')]
    }])
    
    return styled

def format_missing_data(value):
    """Format missing or NaN values as '-'"""
    if pd.isna(value) or value == '' or value is None:
        return '-'
    return value

def safe_format_percentage(value):
    """Safely format percentage values, return '-' for missing data"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return '-'
        return f"{value:.2%}"
    except:
        return '-'

def safe_format_number(value, decimals=2):
    """Safely format numeric values, return '-' for missing data"""
    try:
        if pd.isna(value) or value == '' or value is None:
            return '-'
        return f"{value:.{decimals}f}"
    except:
        return '-'

# Custom CSS for light mode design
st.markdown("""
    <style>
    /* App background & text */
    .stApp {
        background-color: #f9fafb !important;
        color: #111827 !important;
    }
    .main {
        background-color: #ffffff !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5 {
        color: #111827 !important;
        text-shadow: none !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        border-right: 1px solid #e5e7eb;
    }

    /* Inputs */
    input, select, textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 5px !important;
    }

    /* DataFrames */
    div[data-testid="stDataFrame"] table {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb;
    }
    div[data-testid="stDataFrame"] th {
        background-color: #f9fafb !important;
        color: #111827 !important;
    }

    /* Primary buttons */
    button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
    }

    /* Footer */
    footer, div[data-testid="stDecoration"] {
        background: transparent !important;
        color: #6b7280 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 style='text-align: center;'>📊 Vietnamese Stock Market Analysis Platform 📊</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>VN30 Portfolio Analysis & Forecasting System</h3>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Date range
    start_date = st.date_input(
        "Start Date",
        value=datetime(2020, 1, 1),
        min_value=datetime(2019, 1, 1),
        max_value=datetime.now()
    )
    
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now()
    )
    
    st.markdown("---")
    
    # Available stocks from CSV files (matching CSV filenames)
    vn30_stocks = [
        'ACB', 'SHB', 'DGC', 'BID', 'CTG', 'FPT', 'HPG', 'MBB', 'MSN', 'MWG', 'SSI', 'STB', 'VCB', 'VIC', 'VNM', 'SAB', 'VIB', 'VJC', 'PLX', 'VPB', 'LPB', 'VRE', 'HDB', 'BCM', 'VHM', 'GVR', 'TPB', 'TCB', 'SSB', 'GAS'
    ]
    
    selected_stock = st.selectbox(
        "Select Stock for ARIMA Forecast",
        vn30_stocks,
        index=0
    )
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    ) / 100
    
    st.markdown("---")
    st.info("Data Source: Investing.com CSV Files\n\nMarket Index: VNINDEX")

import os
import glob

@st.cache_data
def generate_stock_data(stocks, start_date, end_date):
    """Load Vietnamese stock data from CSV files"""
    
    data = {}
    failed_symbols = []
    
    # Progress bar for data loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data folder path
    data_folder = "D:\Downloads\GÓI 1 CUỐI KỲ\MaiBong\data"
    
    # Get available CSV files
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    csv_symbols = [os.path.basename(f).replace('.csv', '').upper() for f in csv_files]
    
    total_symbols = len(stocks) + 2  # stocks + VNINDEX + VN30
    current_progress = 0
    successful_fetches = 0
    
    # Load VNINDEX
    status_text.text('Loading VNINDEX data...')
    try:
        vnindex_path = os.path.join(data_folder, "vnindex.csv")
        if os.path.exists(vnindex_path):
            vnindex_df = pd.read_csv(vnindex_path)
            vnindex_df = process_csv_data(vnindex_df)
            vnindex_df = filter_data_by_date(vnindex_df, start_date, end_date)
            if not vnindex_df.empty:
                data['VNINDEX'] = vnindex_df['close']
                print("✅ VNINDEX data loaded successfully from CSV")
            else:
                failed_symbols.append('VNINDEX')
                print("⚠️ No VNINDEX data in date range")
        else:
            failed_symbols.append('VNINDEX')
            # st.error("❌ VNINDEX CSV file not found")
    except Exception as e:
        failed_symbols.append('VNINDEX')
        print(f"❌ Failed to load VNINDEX: {str(e)}")
    
    current_progress += 1
    progress_bar.progress(current_progress / total_symbols)
    
    # Load VN30
    status_text.text('Loading VN30 data...')
    try:
        vn30_path = os.path.join(data_folder, "vn30.csv")
        if os.path.exists(vn30_path):
            vn30_df = pd.read_csv(vn30_path)
            vn30_df = process_csv_data(vn30_df)
            vn30_df = filter_data_by_date(vn30_df, start_date, end_date)
            if not vn30_df.empty:
                data['VN30'] = vn30_df['close']
                print("✅ VN30 data loaded successfully from CSV")
            else:
                failed_symbols.append('VN30')
                print("⚠️ No VN30 data in date range")
        else:
            failed_symbols.append('VN30')
            print("❌ VN30 CSV file not found")
    except Exception as e:
        failed_symbols.append('VN30')
        print(f"❌ Failed to load VN30: {str(e)}")
    
    current_progress += 1
    progress_bar.progress(current_progress / total_symbols)
    
    # Load individual stocks
    for i, stock in enumerate(stocks):
        status_text.text(f'Loading {stock} data... ({i+1}/{len(stocks)})')
        
        # Convert stock symbol to CSV filename format
        csv_stock = stock.replace('.VN', '').lower()
        csv_path = os.path.join(data_folder, f"{csv_stock}.csv")
        
        try:
            if os.path.exists(csv_path):
                stock_df = pd.read_csv(csv_path)
                stock_df = process_csv_data(stock_df)
                stock_df = filter_data_by_date(stock_df, start_date, end_date)
                
                if not stock_df.empty and len(stock_df) > 10:
                    data[csv_stock.upper()] = stock_df['close']
                    successful_fetches += 1
                else:
                    failed_symbols.append(stock)
                    if len(failed_symbols) <= 5:
                        st.warning(f"⚠️ Insufficient data for {stock}")
            else:
                failed_symbols.append(stock)
                if len(failed_symbols) <= 5:
                    st.warning(f"⚠️ CSV file not found for {stock}")
                
        except Exception as e:
            failed_symbols.append(stock)
            if len(failed_symbols) <= 5:
                st.warning(f"⚠️ Could not load data for {stock}: {str(e)}")
        
        current_progress += 1
        progress_bar.progress(current_progress / total_symbols)
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create DataFrame from successfully fetched data
    if not data:
        st.error("❌ No data could be loaded from CSV files. Please check the data folder.")
        return None
    
    df = pd.DataFrame(data)
    
    # Forward fill missing values and clean up
    df = df.ffill()
    df = df.dropna(how='all')
    
    # Success message with statistics
    real_stocks = successful_fetches
    total_stocks = len(stocks)
    success_rate = (real_stocks / total_stocks) * 100 if total_stocks > 0 else 0
    
    print(f"✅ Successfully loaded {len(data)} symbols with {len(df)} trading days from CSV files")
    print(f"📊 Stock success rate: {real_stocks}/{total_stocks} ({success_rate:.1f}%)")
    
    if failed_symbols:
        failed_count = len(failed_symbols)
        if failed_count <= 10:
            st.warning(f"❌ Failed to load: {', '.join(failed_symbols[:10])}")
        else:
            st.warning(f"❌ Failed to load {failed_count} symbols: {', '.join(failed_symbols[:5])} and {failed_count-5} more...")
        
        st.info("💡 Tips:")
        st.info("- Make sure CSV files exist in the data folder")
        st.info("- Check date ranges in your CSV data")
        st.info("- Data source: Investing.com CSV files")
    
    return df

def process_csv_data(df):
    """Process CSV data from Investing.com format to pandas DataFrame"""
    # Remove BOM if present
    if df.columns[0].startswith('\ufeff'):
        df.columns = [df.columns[0].replace('\ufeff', '')] + list(df.columns[1:])
    
    # Rename columns to English
    column_mapping = {
        'Ngày': 'Date',
        'Lần cuối': 'Close',
        'Mở': 'Open', 
        'Cao': 'High',
        'Thấp': 'Low',
        'KL': 'Volume',
        '% Thay đổi': 'Change%'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean and convert data
    try:
        # Convert date
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        # Clean and convert numeric columns
        numeric_cols = ['Close', 'Open', 'High', 'Low']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert to float
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Set date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Rename close column for consistency
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
            
    except Exception as e:
        st.error(f"Error processing CSV data: {str(e)}")
        return pd.DataFrame()
    
    return df

def filter_data_by_date(df, start_date, end_date):
    """Filter DataFrame by date range"""
    if df.empty:
        return df
        
    try:
        # Convert start_date and end_date to pandas datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter data
        filtered_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return filtered_df
    except Exception:
        return df

# Calculate returns
def calculate_returns(df):
    """Calculate daily and monthly returns"""
    daily_returns = df.pct_change().dropna()
    
    # Monthly returns
    monthly_df = df.resample('M').last()
    monthly_returns = monthly_df.pct_change().dropna()
    
    return daily_returns, monthly_returns

# CAPM Beta calculation
def calculate_beta(stock_returns, market_returns):
    """Calculate beta using CAPM"""
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    
    # Also calculate R-squared
    correlation = np.corrcoef(stock_returns, market_returns)[0][1]
    r_squared = correlation ** 2
    
    return beta, r_squared

# ARIMA forecast
def arima_forecast(data, periods=30):
    """Forecast using ARIMA model"""
    try:
        model = ARIMA(data, order=(5, 1, 0))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=periods)
        
        # Confidence intervals
        forecast_df = fitted_model.get_forecast(steps=periods)
        conf_int = forecast_df.conf_int()
        
        return forecast, conf_int, fitted_model
    except:
        return None, None, None

# Portfolio optimization
def create_portfolios(betas, returns, stocks):
    """Create stable and risky portfolios"""
    beta_df = pd.DataFrame({
        'Stock': stocks,
        'Beta': betas,
        'Avg_Return': returns
    }).sort_values('Beta')
    
    # Stable portfolio (low beta stocks)
    stable_stocks = beta_df.nsmallest(5, 'Beta')
    stable_portfolio = stable_stocks.copy()
    stable_portfolio['Weight'] = 1 / len(stable_portfolio)
    
    # Risky portfolio (high beta stocks)
    risky_stocks = beta_df.nlargest(5, 'Beta')
    risky_portfolio = risky_stocks.copy()
    risky_portfolio['Weight'] = 1 / len(risky_portfolio)
    
    return stable_portfolio, risky_portfolio, beta_df

# Add Run Analysis button in sidebar
with st.sidebar:
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if not run_analysis:
        st.info("👆 Click 'Run Analysis' to fetch data and start the analysis")

# Only load data when button is clicked
if run_analysis:
    with st.spinner("Loading and processing data from CSV files..."):
        df = generate_stock_data(vn30_stocks, start_date, end_date)
        
        if df is None or df.empty:
            st.error("Failed to load data. Please check your internet connection or try a different date range.")
            st.stop()
        
        daily_returns, monthly_returns = calculate_returns(df)
else:
    # Show placeholder content when analysis hasn't been run
    st.stop()

# Data Overview Section
st.header("Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stocks", len(vn30_stocks), "VN30")
with col2:
    st.metric("Trading Days", len(df), f"{len(df)/365:.1f} years")
with col3:
    st.metric("Date Range", f"{(end_date - start_date).days} days")
with col4:
    st.metric("Market Index", "VNINDEX", "Benchmark")

# Market overview chart
st.header(f"Chart Overview")

# Lấy dữ liệu VN30 index trực tiếp từ cột VN30
vn30_index = df['VN30'] if 'VN30' in df.columns else df['VNINDEX']

fig_market = go.Figure()

# VNINDEX
fig_market.add_trace(go.Scatter(
    x=df.index,
    y=df['VNINDEX'],
    mode='lines',
    name='VNINDEX',
    line=dict(color='red', width=2)
))

# VN30 Index
fig_market.add_trace(go.Scatter(
    x=df.index,
    y=vn30_index,
    mode='lines',
    name='VN30 Index',
    line=dict(color='#f5a623', width=2)  # màu vàng, nét đứt
))

fig_market.update_layout(
    title="VNINDEX vs VN30 Index",
    xaxis_title="Date",
    yaxis_title="Index Value",
    hovermode='x unified',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=400
)

st.plotly_chart(fig_market, use_container_width=True)

# Stock comparison
fig_stocks = go.Figure()
colors = px.colors.qualitative.Set3

# Only plot stocks that are actually in the DataFrame
available_stocks = [stock for stock in vn30_stocks if stock in df.columns]
for i, stock in enumerate(available_stocks[:10]):
    fig_stocks.add_trace(go.Scatter(
        x=df.index,
        y=df[stock],
        mode='lines',
        name=stock,
        line=dict(color=colors[i % len(colors)], width=1.5)
    ))

fig_stocks.update_layout(
    title="Stock Prices Chart",
    xaxis_title="Date",
    yaxis_title="Price (VND)",
    hovermode='x unified',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_stocks, use_container_width=True)

# Returns Analysis
st.header("Returns Analysis")

col1, col2 = st.columns(2)

with col1:
    # Calculate market daily returns
    market_daily_returns = daily_returns['VNINDEX']

    # Tạo histogram
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Histogram(
        x=market_daily_returns,
        nbinsx=50,
        name='VNINDEX Daily Returns',
        marker_color='#667eea',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.7,
        histnorm='probability density'  # Chuẩn hóa theo mật độ xác suất
    ))

    # Tính và vẽ đường Normal Distribution
    mean = market_daily_returns.mean()
    std = market_daily_returns.std()
    x_range = np.linspace(market_daily_returns.min(), market_daily_returns.max(), 200)
    y_normal = stats.norm.pdf(x_range, mean, std)
    fig_daily.add_trace(go.Scatter(
        x=x_range,
        y=y_normal,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))

    fig_daily.update_layout(
        title=dict(
            text="Daily Returns Distribution (VNINDEX)",
            x=0.5,
            xanchor='center',
            yanchor='top',
            pad=dict(b=5)
        ),
        xaxis_title="Daily Returns",
        yaxis_title="Density",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350,
        showlegend=True
    )

    st.plotly_chart(fig_daily, use_container_width=True)


    market_monthly_returns = monthly_returns['VNINDEX']

    # Tạo histogram
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Histogram(
        x=market_monthly_returns,
        nbinsx=30,
        name='VNINDEX Monthly Returns',
        marker_color='#764ba2',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.7,
        histnorm='probability density'  # Chuẩn hóa theo mật độ xác suất
    ))

    # Tính và vẽ đường Normal Distribution
    mean_m = market_monthly_returns.mean()
    std_m = market_monthly_returns.std()
    x_range_m = np.linspace(market_monthly_returns.min(), market_monthly_returns.max(), 200)
    y_normal_m = stats.norm.pdf(x_range_m, mean_m, std_m)
    fig_monthly.add_trace(go.Scatter(
        x=x_range_m,
        y=y_normal_m,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))

with col2:    
    # Returns statistics
    st.subheader("📊 Returns Statistics")

    stats_data = {
        'Metric': ['Mean Daily Return', 'Std Daily Return', 'Mean Monthly Return', 'Std Monthly Return', 
                    'Sharpe Ratio (Daily)', 'Max Drawdown'],
        'VNINDEX': [
            safe_format_percentage(market_daily_returns.mean()) if not market_daily_returns.empty else '-',
            safe_format_percentage(market_daily_returns.std()) if not market_daily_returns.empty else '-',
            safe_format_percentage(market_monthly_returns.mean()) if not market_monthly_returns.empty else '-',
            safe_format_percentage(market_monthly_returns.std()) if not market_monthly_returns.empty else '-',
            safe_format_number((market_daily_returns.mean() - risk_free_rate/252) / market_daily_returns.std(), 4) if not market_daily_returns.empty else '-',
            safe_format_percentage(((df['VNINDEX'] / df['VNINDEX'].cummax() - 1).min())) if 'VNINDEX' in df.columns and not df['VNINDEX'].empty else '-'
        ]
    }

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ARIMA Forecast Section
st.header(f"ARIMA Forecast for {selected_stock}")

if selected_stock in df.columns:
    stock_prices = df[selected_stock]
    forecast, conf_int, model = arima_forecast(stock_prices.values, periods=30)
else:
    st.error(f"❌ {selected_stock} data not available in the dataset")
    forecast, conf_int, model = None, None, None

if forecast is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Forecast chart
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=df.index[-90:],
            y=stock_prices[-90:],
            mode='lines',
            name='Historical',
            line=dict(color='#667eea', width=2)
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='#f56565', width=2, dash='dash')
        ))
        
        # Confidence interval
        if isinstance(conf_int, pd.DataFrame):
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
        elif isinstance(conf_int, np.ndarray):
            # Nếu là ndarray, giả sử cột 0 là lower và cột 1 là upper
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]
        else:
            lower = upper = np.array([np.nan] * len(forecast_dates))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(245, 101, 101, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence',
            showlegend=True
        ))
        
        fig_forecast.update_layout(
            title=f"{selected_stock} - 30 Days Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (VND)",
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
    with col2:
        st.subheader("Forecast Metrics")
        
        forecast_return = (forecast[-1] - stock_prices.iloc[-1]) / stock_prices.iloc[-1]
        
        current_price = safe_format_number(stock_prices.iloc[-1], 2) if not stock_prices.empty else '-'
        forecast_price = safe_format_number(forecast[-1], 2) if forecast is not None and len(forecast) > 0 else '-'
        expected_return = safe_format_percentage(forecast_return) if forecast is not None else '-'
        
        st.metric("Current Price", f"{current_price} VND")
        st.metric("Forecasted Price (30d)", f"{forecast_price} VND")
        st.metric("Expected Return", expected_return, 
                    delta=expected_return if expected_return != '-' else None)
        
        st.info(f"**Model:** ARIMA(5,1,0)\n\n**AIC:** {model.aic:.2f}\n\n**BIC:** {model.bic:.2f}")
else:
    st.error("Failed to generate forecast. Please try different parameters.")

# === Detailed Forecast Results Table ===
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Forecasted Price": forecast,
    "Lower CI": lower,
    "Upper CI": upper
})

# Tính % thay đổi so với giá hiện tại
forecast_df["Daily % Change"] = forecast_df["Forecasted Price"].pct_change() * 100
forecast_df["Cumulative % Change"] = (forecast_df["Forecasted Price"] / stock_prices.iloc[-1] - 1) * 100

# Format đẹp để hiển thị
forecast_display = forecast_df.copy()
forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m-%d")
forecast_display["Forecasted Price"] = forecast_display["Forecasted Price"].apply(lambda x: safe_format_number(x, 2) if pd.notna(x) else '-')
forecast_display["Lower CI"] = forecast_display["Lower CI"].apply(lambda x: safe_format_number(x, 2) if pd.notna(x) else '-')
forecast_display["Upper CI"] = forecast_display["Upper CI"].apply(lambda x: safe_format_number(x, 2) if pd.notna(x) else '-')
forecast_display["Daily % Change"] = forecast_display["Daily % Change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')
forecast_display["Cumulative % Change"] = forecast_display["Cumulative % Change"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else '-')

st.subheader("\n\n\nDetailed Forecast Results")

st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True
)

# === Nút tải xuống bảng dự báo ===
csv_forecast = forecast_display.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Download Forecast Results (CSV)",
    data=csv_forecast,
    file_name=f"{selected_stock}_forecast_results_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# CAPM Beta Analysis
st.header("CAPM Beta Analysis")

# Calculate betas for all stocks that are available
market_returns = daily_returns['VNINDEX'] if 'VNINDEX' in daily_returns.columns else daily_returns.iloc[:, 0]
betas = []
r_squareds = []
avg_returns = []
available_stocks_for_beta = []

for stock in vn30_stocks:
    if stock in daily_returns.columns:
        stock_returns = daily_returns[stock]
        beta, r2 = calculate_beta(stock_returns, market_returns)
        betas.append(beta)
        r_squareds.append(r2)
        avg_returns.append(stock_returns.mean())
        available_stocks_for_beta.append(stock)

# Beta visualization
col1, col2, col3 = st.columns(3)

with col1:
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Bar(
        x=available_stocks_for_beta,
        y=betas,
        marker_color=['#48bb78' if b < 1 else '#f56565' for b in betas],
        text=[f"{b:.2f}" for b in betas],
        textposition='outside'
    ))
    
    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                        annotation_text="Market Beta = 1.0")
    
    fig_beta.update_layout(
        title="Beta Distribution across VN30",
        xaxis_title="Stock",
        yaxis_title="Beta",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_beta, use_container_width=True)

with col2:    
    fig_risk_return = go.Figure()
    
    fig_risk_return.add_trace(go.Scatter(
        x=betas,
        y=[r * 252 for r in avg_returns],  # Annualized returns
        mode='markers+text',
        text=available_stocks_for_beta,
        textposition='top center',
        marker=dict(
            size=12,
            color=betas,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Beta")
        ),
        name='Stocks'
    ))
    
    fig_risk_return.update_layout(
        title="Risk-Return Trade-off",
        xaxis_title="Beta (Systematic Risk)",
        yaxis_title="Annualized Return",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_risk_return, use_container_width=True)

with col3:
    
    # Lấy dữ liệu lợi nhuận hàng ngày của cổ phiếu và VNINDEX
    stock_daily_returns = daily_returns[selected_stock]
    market_daily_returns = daily_returns['VNINDEX']

    # Vẽ scatter plot
    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=market_daily_returns,
        y=stock_daily_returns,
        mode='markers',
        marker=dict(
            color=np.where(stock_daily_returns >= 0, '#00FF00', '#FF4136'),  # Xanh nếu tăng, đỏ nếu giảm
            opacity=0.7,
            size=6,
            line=dict(width=0.5, color='white')
        ),
        name=f"{selected_stock} vs VNINDEX"
    ))

    # Vẽ đường hồi quy tuyến tính
    slope, intercept, r_value, p_value, std_err = stats.linregress(market_daily_returns, stock_daily_returns)
    x_vals = np.linspace(market_daily_returns.min(), market_daily_returns.max(), 100)
    y_vals = intercept + slope * x_vals

    fig_scatter.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='yellow', width=2),
        name=f"Regression line (β={slope:.2f})"
    ))

    fig_scatter.update_layout(
        title=dict(
            text=f"Scatter Plot: {selected_stock} vs VNINDEX",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="VNINDEX Daily Returns",
        yaxis_title=f"{selected_stock} Daily Returns",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=350,
        showlegend=True
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# Beta table
st.subheader("Detailed Beta Analysis")

beta_table = pd.DataFrame({
    'Stock': available_stocks_for_beta,
    'Beta': [safe_format_number(b, 3) for b in betas],
    'R²': [safe_format_number(r, 3) for r in r_squareds],
    'Avg Daily Return': [safe_format_percentage(r) for r in avg_returns],
    'Annualized Return': [safe_format_percentage(r * 252) for r in avg_returns],
    'Risk Level': ['Low' if b < 0.8 else 'Medium' if b < 1.2 else 'High' for b in betas]
})

beta_table = beta_table.sort_values('Beta')

styled_table = (
    beta_table.style
    .set_properties(**{
        'text-align': 'left'
    })
)

# Căn giữa cả header
styled_table.set_table_styles([{
    'selector': 'th',
    'props': [('text-align', 'left')]
}])

st.dataframe(styled_table, use_container_width=True, hide_index=True)



# Portfolio Construction
st.header("Portfolio Construction & Recommendations")

stable_portfolio, risky_portfolio, all_stocks = create_portfolios(
    betas, avg_returns, available_stocks_for_beta
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🛡️ Stable Portfolio (Low Beta)")
    
    # Portfolio metrics
    stable_beta = (stable_portfolio['Beta'] * stable_portfolio['Weight']).sum()
    stable_return = (stable_portfolio['Avg_Return'] * stable_portfolio['Weight']).sum() * 252
    
    st.metric("Portfolio Beta", safe_format_number(stable_beta, 3))
    st.metric("Expected Annual Return", safe_format_percentage(stable_return))
    
    # Pie chart
    fig_stable = go.Figure(data=[go.Pie(
        labels=stable_portfolio['Stock'],
        values=stable_portfolio['Weight'],
        hole=.4,
        marker_colors=px.colors.sequential.Blues_r
    )])
    
    fig_stable.update_layout(
        title="Portfolio Allocation",
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)'  
    )
    
    st.plotly_chart(fig_stable, use_container_width=True)
    
    # Table
    stable_display = stable_portfolio[['Stock', 'Beta', 'Avg_Return', 'Weight']].copy()
    stable_display['Weight'] = stable_display['Weight'].apply(lambda x: safe_format_percentage(x))
    stable_display['Avg_Return'] = stable_display['Avg_Return'].apply(lambda x: safe_format_percentage(x*252))
    stable_display['Beta'] = stable_display['Beta'].apply(lambda x: safe_format_number(x, 3))
    
    st.dataframe(stable_display, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🚀 Risky Portfolio (High Beta)")
    
    # Portfolio metrics
    risky_beta = (risky_portfolio['Beta'] * risky_portfolio['Weight']).sum()
    risky_return = (risky_portfolio['Avg_Return'] * risky_portfolio['Weight']).sum() * 252
    
    st.metric("Portfolio Beta", safe_format_number(risky_beta, 3))
    st.metric("Expected Annual Return", safe_format_percentage(risky_return))
    
    # Pie chart
    fig_risky = go.Figure(data=[go.Pie(
        labels=risky_portfolio['Stock'],
        values=risky_portfolio['Weight'],
        hole=.4,
        marker_colors=px.colors.sequential.Reds
    )])
    
    fig_risky.update_layout(
        title="Portfolio Allocation",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_risky, use_container_width=True)
    
    # Table
    risky_display = risky_portfolio[['Stock', 'Beta', 'Avg_Return', 'Weight']].copy()
    risky_display['Weight'] = risky_display['Weight'].apply(lambda x: safe_format_percentage(x))
    risky_display['Avg_Return'] = risky_display['Avg_Return'].apply(lambda x: safe_format_percentage(x*252))
    risky_display['Beta'] = risky_display['Beta'].apply(lambda x: safe_format_number(x, 3))
    
    st.dataframe(risky_display, use_container_width=True, hide_index=True)

# Portfolio comparison
st.subheader("Portfolio Comparison")

comparison_data = {
    'Metric': ['Portfolio Beta', 'Expected Annual Return', 'Risk Level', 'Volatility', 
                'Sharpe Ratio (Est.)', 'Recommended For'],
    'Stable Portfolio': [
        safe_format_number(stable_beta, 3),
        safe_format_percentage(stable_return),
        "Low",
        safe_format_number(stable_portfolio['Beta'].std(), 3),
        safe_format_number((stable_return - risk_free_rate) / stable_portfolio['Beta'].std(), 3),
        "Conservative investors, Capital preservation"
    ],
    'Risky Portfolio': [
        safe_format_number(risky_beta, 3),
        safe_format_percentage(risky_return),
        "High",
        safe_format_number(risky_portfolio['Beta'].std(), 3),
        safe_format_number((risky_return - risk_free_rate) / risky_portfolio['Beta'].std(), 3),
        "Aggressive investors, Growth seeking"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)