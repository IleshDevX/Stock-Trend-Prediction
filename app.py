import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Page config and styling
st.set_page_config(
    page_title="Trend Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .company-header {
        background: linear-gradient(90deg, #1F2937 0%, #374151 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #374151;
    }
    
    .news-item {
        background-color: #1F2937;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid #3B82F6;
    }
    
    .sentiment-positive {
        border-left-color: #10B981 !important;
    }
    
    .sentiment-negative {
        border-left-color: #EF4444 !important;
    }
    
    .sentiment-neutral {
        border-left-color: #6B7280 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #0F172A;
        border-radius: 12px;
        padding: 6px;
        margin-bottom: 2rem;
        border: 1px solid #1E293B;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #64748B;
        font-size: 14px;
        font-weight: 600;
        border: none;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 12px 20px;
        margin: 0;
        text-transform: none;
        letter-spacing: 0.025em;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E40AF !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important;
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #1E293B;
        color: #94A3B8;
        transform: translateY(-0.5px);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px);
    }
    
    /* Tab panel styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker_symbol, start_date, end_date):
    """Fetch stock data with caching"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return None, f"No data found for ticker '{ticker_symbol}'"
        
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_company_info(ticker_symbol):
    """Fetch company information with caching"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        return info
    except:
        return {}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_news(ticker_symbol):
    """Fetch news with caching"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        return news[:3]  # Get latest 3 news items
    except:
        return []

@st.cache_resource
def load_prediction_model():
    """Load the LSTM model with caching and error handling"""
    try:
        model = load_model('keras_model.h5')
        st.sidebar.success("✅ LSTM Model Loaded")
        return model
    except Exception as e:
        st.sidebar.warning("⚠️ LSTM Model not available")
        st.sidebar.info("📊 Using trend analysis instead")
        return None

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        return 'Positive', scores['compound']
    elif scores['compound'] <= -0.05:
        return 'Negative', scores['compound']
    else:
        return 'Neutral', scores['compound']

def assess_stock_compatibility(df, ticker_symbol):
    """Assess how well the stock works with our prediction models"""
    try:
        # Calculate metrics
        avg_volume = df['Volume'].mean()
        price_volatility = df['Close'].pct_change().std() * 100
        data_length = len(df)
        price_range = df['Close'].max() - df['Close'].min()
        avg_price = df['Close'].mean()
        
        score = 0
        feedback = []
        
        # Volume check (higher is better for LSTM)
        if avg_volume > 10_000_000:
            score += 3
            feedback.append("✅ High trading volume - excellent for LSTM")
        elif avg_volume > 1_000_000:
            score += 2
            feedback.append("🟡 Good trading volume")
        else:
            score += 1
            feedback.append("🔴 Low trading volume - may affect accuracy")
        
        # Volatility check (moderate is best)
        if 1 <= price_volatility <= 3:
            score += 3
            feedback.append("✅ Optimal volatility for predictions")
        elif price_volatility <= 5:
            score += 2
            feedback.append("🟡 Moderate volatility")
        else:
            score += 1
            feedback.append("🔴 High volatility - predictions less reliable")
        
        # Data length check
        if data_length >= 1000:
            score += 3
            feedback.append("✅ Sufficient historical data")
        elif data_length >= 500:
            score += 2
            feedback.append("🟡 Adequate historical data")
        else:
            score += 1
            feedback.append("🔴 Limited historical data")
        
        # Price stability check
        if price_range / avg_price < 2:
            score += 2
            feedback.append("✅ Stable price range")
        else:
            score += 1
            feedback.append("🟡 Wide price range detected")
        
        # Overall assessment
        if score >= 10:
            compatibility = "🟢 Excellent"
        elif score >= 8:
            compatibility = "🟡 Good"
        elif score >= 6:
            compatibility = "🟠 Fair"
        else:
            compatibility = "🔴 Poor"
        
        return compatibility, feedback, score
        
    except Exception as e:
        return "❓ Unknown", [f"Error assessing compatibility: {str(e)}"], 0

def create_candlestick_chart(df, ma_50=False, ma_200=False, predictions=None, pred_dates=None):
    """Create interactive candlestick chart with Plotly"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price & Volume', 'Volume'),
        row_width=[0.2, 0.7]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC",
            increasing_line_color='#10B981',
            decreasing_line_color='#EF4444'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if ma_50:
        ma_50_data = df['Close'].rolling(50).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=ma_50_data, name='50 MA', line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    if ma_200:
        ma_200_data = df['Close'].rolling(200).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=ma_200_data, name='200 MA', line=dict(color='purple', width=1)),
            row=1, col=1
        )
    
    # Predictions
    if predictions is not None and pred_dates is not None:
        fig.add_trace(
            go.Scatter(
                x=pred_dates, 
                y=predictions.flatten(), 
                name='Prediction',
                line=dict(color='#3B82F6', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['#10B981' if close >= open else '#EF4444' 
              for close, open in zip(df['Close'], df['Open'])]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Stock Price Analysis",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        showlegend=True
    )
    
    return fig

def predict_prices(df, model, days_to_predict=30):
    """Make price predictions using dynamic methods"""
    if model is None or len(df) < 100:
        # Fallback to simple trend analysis if model fails or insufficient data
        return predict_with_trend_analysis(df, days_to_predict)
    
    try:
        # Dynamic approach: Use percentage changes instead of absolute prices
        returns = df['Close'].pct_change().dropna()
        
        # Check if we have enough data
        if len(returns) < 100:
            return predict_with_trend_analysis(df, days_to_predict)
        
        # Initialize and fit the scaler on returns
        scaler = MinMaxScaler(feature_range=(0,1))
        
        # Use returns instead of absolute prices for better generalization
        returns_array = returns.values.reshape(-1, 1)
        scaled_returns = scaler.fit_transform(returns_array)
        
        # Prepare sequences for LSTM
        sequence_length = min(60, len(scaled_returns) - 1)  # Dynamic sequence length
        x_test = []
        
        # Use the last sequence for prediction
        last_sequence = scaled_returns[-sequence_length:]
        x_test.append(last_sequence)
        x_test = np.array(x_test)
        
        # Reshape for LSTM if needed
        if len(x_test.shape) == 2:
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
        # Make predictions for returns
        predicted_returns = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_to_predict):
            # Predict next return
            pred_return = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)
            predicted_returns.append(pred_return[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred_return[0, 0]
        
        # Inverse transform returns
        predicted_returns = np.array(predicted_returns).reshape(-1, 1)
        predicted_returns = scaler.inverse_transform(predicted_returns)
        
        # Convert returns back to prices
        last_price = df['Close'].iloc[-1]
        predicted_prices = []
        current_price = last_price
        
        for ret in predicted_returns.flatten():
            current_price = current_price * (1 + ret)
            predicted_prices.append(current_price)
        
        predicted_prices = np.array(predicted_prices).reshape(-1, 1)
        
        # Create prediction dates
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predicted_prices))
        
        return predicted_prices, pred_dates
        
    except Exception as e:
        # Fallback to trend analysis if LSTM fails
        st.warning(f"LSTM prediction failed: {str(e)}. Using trend analysis instead.")
        return predict_with_trend_analysis(df, days_to_predict)

def predict_with_trend_analysis(df, days_to_predict=30):
    """Fallback prediction using trend analysis and moving averages"""
    try:
        # Calculate various indicators
        close_prices = df['Close']
        
        # Short and long term moving averages
        ma_short = close_prices.rolling(window=10).mean().iloc[-1]
        ma_long = close_prices.rolling(window=30).mean().iloc[-1]
        
        # Recent trend (last 20 days)
        recent_prices = close_prices.tail(20)
        trend_slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
        
        # Volatility (standard deviation of recent returns)
        recent_returns = close_prices.pct_change().tail(30).std()
        
        # Base prediction on trend and momentum
        last_price = close_prices.iloc[-1]
        
        # Determine trend direction
        if ma_short > ma_long:
            trend_direction = 1  # Upward trend
        else:
            trend_direction = -1  # Downward trend
        
        # Generate predictions
        predicted_prices = []
        for i in range(days_to_predict):
            # Combine trend with some randomness based on volatility
            daily_change = trend_slope * trend_direction * (1 + np.random.normal(0, recent_returns * 0.5))
            
            if i == 0:
                next_price = last_price + daily_change
            else:
                next_price = predicted_prices[-1] + daily_change
            
            # Add mean reversion component
            if abs(next_price - last_price) / last_price > 0.1:  # If change > 10%
                next_price = next_price * 0.9 + last_price * 0.1  # Pull back towards original
            
            predicted_prices.append(next_price)
        
        predicted_prices = np.array(predicted_prices).reshape(-1, 1)
        
        # Create prediction dates
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predicted_prices))
        
        return predicted_prices, pred_dates
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

# Sidebar
st.sidebar.markdown("# 📈 Trend Forecaster")
st.sidebar.markdown("---")

# Stock selection
st.sidebar.markdown("### Stock Selection")
ticker_symbol = st.sidebar.text_input(
    "Enter Stock Ticker", 
    value="AAPL",
    help="Enter a valid stock ticker (e.g., AAPL, GOOGL, MSFT)"
).upper()

# Recommended stocks section
with st.sidebar.expander("📋 Recommended Stocks"):
    st.markdown("**🔥 Popular Tech Stocks:**")
    recommended_stocks = {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.", 
        "MSFT": "Microsoft Corp.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms"
    }
    
    st.markdown("**💰 Blue Chip Stocks:**")
    blue_chip_stocks = {
        "JPM": "JPMorgan Chase",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc.",
        "PG": "Procter & Gamble",
        "HD": "Home Depot",
        "UNH": "UnitedHealth",
        "DIS": "Walt Disney"
    }
    
    st.markdown("**🏦 Financial Sector:**")
    financial_stocks = {
        "BAC": "Bank of America",
        "WFC": "Wells Fargo",
        "GS": "Goldman Sachs",
        "C": "Citigroup",
        "AXP": "American Express"
    }
    
    st.markdown("**🎯 Model Works Best With:**")
    st.write("• High volume stocks (>1M daily)")
    st.write("• Established companies (5+ years)")
    st.write("• Regular trading patterns")
    st.write("• Minimal gaps/splits")
    
    # Quick select buttons for recommended stocks
    st.markdown("**Quick Select:**")
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        if st.button("📱 AAPL", key="aapl_btn"):
            ticker_symbol = "AAPL"
            st.rerun()
        if st.button("🔍 GOOGL", key="googl_btn"):
            ticker_symbol = "GOOGL"
            st.rerun()
        if st.button("💻 MSFT", key="msft_btn"):
            ticker_symbol = "MSFT"
            st.rerun()
    
    with quick_col2:
        if st.button("🚗 TSLA", key="tsla_btn"):
            ticker_symbol = "TSLA"
            st.rerun()
        if st.button("🛒 AMZN", key="amzn_btn"):
            ticker_symbol = "AMZN"
            st.rerun()
        if st.button("🎯 NVDA", key="nvda_btn"):
            ticker_symbol = "NVDA"
            st.rerun()

# Date range selection
st.sidebar.markdown("### Date Range")
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date = st.date_input(
        "Start Date",
        value=date(2020, 1, 1),
        max_value=date.today()
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=date.today(),
        max_value=date.today()
    )

# Analysis settings
st.sidebar.markdown("### Analysis Settings")
days_to_predict = st.sidebar.slider(
    "Prediction Period (days)",
    min_value=1,
    max_value=90,
    value=30
)

# Chart options
st.sidebar.markdown("### Chart Options")
show_ma_50 = st.sidebar.checkbox("Show 50-day MA", value=True)
show_ma_200 = st.sidebar.checkbox("Show 200-day MA", value=True)

# Analysis button
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.rerun()

# Disclaimer
with st.sidebar.expander("⚠️ Disclaimer"):
    st.write("""
    This tool is for educational purposes only. 
    Stock predictions are not financial advice. 
    Always consult with a financial advisor before making investment decisions.
    """)

# Main content area
if ticker_symbol:
    # Fetch data
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        df, error = fetch_stock_data(ticker_symbol, start_date, end_date)
        company_info = fetch_company_info(ticker_symbol)
        
    if error:
        st.error(f"Error: {error}")
        st.stop()
    
    if df is None or df.empty:
        st.error("No data available for the selected date range.")
        st.stop()
    
    # Company header
    company_name = company_info.get('longName', ticker_symbol)
    current_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
    price_change_pct = (price_change / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
    
    st.markdown(f"""
    <div class="company-header">
        <h1 style="margin:0; color: white;">{company_name} ({ticker_symbol})</h1>
        <h2 style="margin:0; color: #3B82F6;">${current_price:.2f} 
            <span style="color: {'#10B981' if price_change >= 0 else '#EF4444'};">
                {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change_pct:.2f}%)
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock compatibility assessment
    compatibility, feedback, score = assess_stock_compatibility(df, ticker_symbol)
    
    col_comp1, col_comp2 = st.columns([3, 1])
    with col_comp1:
        with st.expander(f"📊 Model Compatibility: {compatibility}", expanded=False):
            st.write(f"**Compatibility Score: {score}/11**")
            for item in feedback:
                st.write(item)
    with col_comp2:
        st.metric("Compatibility", compatibility.split()[1] if len(compatibility.split()) > 1 else "Unknown")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        market_cap = company_info.get('marketCap', 0)
        if market_cap > 1e12:
            market_cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap > 1e9:
            market_cap_str = f"${market_cap/1e9:.2f}B"
        elif market_cap > 1e6:
            market_cap_str = f"${market_cap/1e6:.2f}M"
        else:
            market_cap_str = "N/A"
        st.metric("Market Cap", market_cap_str)
    
    with col2:
        pe_ratio = company_info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            pe_ratio = f"{pe_ratio:.2f}"
        st.metric("P/E Ratio", pe_ratio)
    
    with col3:
        week_52_high = company_info.get('fiftyTwoWeekHigh', 'N/A')
        if isinstance(week_52_high, (int, float)):
            week_52_high = f"${week_52_high:.2f}"
        st.metric("52W High", week_52_high)
    
    with col4:
        week_52_low = company_info.get('fiftyTwoWeekLow', 'N/A')
        if isinstance(week_52_low, (int, float)):
            week_52_low = f"${week_52_low:.2f}"
        st.metric("52W Low", week_52_low)
    
    with col5:
        sector = company_info.get('sector', 'N/A')
        st.metric("Sector", sector)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Price Analysis & Forecast", "🏢 Company Profile", "📰 News & Sentiment"])
    
    with tab1:
        # Load model and make predictions
        model = load_prediction_model()
        
        # Always try to make predictions (will fallback to trend analysis if needed)
        with st.spinner("Generating price predictions..."):
            predictions, pred_dates = predict_prices(df, model, days_to_predict)
        
        # Show prediction method info
        col_info1, col_info2 = st.columns([3, 1])
        with col_info2:
            if model is not None:
                st.success("🤖 LSTM Model")
            else:
                st.info("📈 Trend Analysis")
        
        # Create and display chart
        fig = create_candlestick_chart(df, show_ma_50, show_ma_200, predictions, pred_dates)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction summary
        if predictions is not None:
            with st.expander("📈 Prediction Summary"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Predicted High", f"${np.max(predictions):.2f}")
                with col2:
                    st.metric("Predicted Low", f"${np.min(predictions):.2f}")
                with col3:
                    predicted_return = (predictions[-1] - current_price) / current_price * 100
                    st.metric("Expected Return", f"{predicted_return[0]:.2f}%")
                with col4:
                    prediction_method = "LSTM Neural Network" if model else "Technical Analysis"
                    st.metric("Method", prediction_method)
                
                # Additional insights
                st.markdown("#### 💡 Prediction Insights")
                
                avg_prediction = np.mean(predictions)
                volatility = np.std(predictions) / avg_prediction * 100
                
                if model:
                    st.write("🔹 **LSTM Model**: Using deep learning to analyze historical patterns")
                else:
                    st.write("🔹 **Trend Analysis**: Using moving averages and momentum indicators")
                
                st.write(f"🔹 **Average Predicted Price**: ${avg_prediction:.2f}")
                st.write(f"🔹 **Prediction Volatility**: {volatility:.1f}%")
                
                if volatility > 15:
                    st.warning("⚠️ High volatility detected - predictions may be less reliable")
                elif volatility < 5:
                    st.success("✅ Low volatility - more stable price predictions")
        else:
            st.error("❌ Unable to generate predictions. Please try a different stock or date range.")
    
    with tab2:
        # Company profile
        st.markdown("### Business Summary")
        business_summary = company_info.get('longBusinessSummary', 'No business summary available.')
        st.write(business_summary)
        
        # Key executives
        st.markdown("### Key Executives")
        officers = company_info.get('companyOfficers', [])
        if officers:
            exec_data = []
            for officer in officers[:4]:  # Top 4 executives
                exec_data.append({
                    'Name': officer.get('name', 'N/A'),
                    'Title': officer.get('title', 'N/A'),
                    'Age': officer.get('age', 'N/A')
                })
            exec_df = pd.DataFrame(exec_data)
            st.dataframe(exec_df, use_container_width=True)
        else:
            st.write("No executive information available.")
        
        # Financial ratios
        st.markdown("### Key Financial Ratios")
        ratios_data = {
            'Metric': ['Price to Book', 'EPS (TTM)', 'Dividend Yield', 'Debt to Equity', 'ROE'],
            'Value': [
                f"{company_info.get('priceToBook', 'N/A'):.2f}" if isinstance(company_info.get('priceToBook'), (int, float)) else 'N/A',
                f"${company_info.get('trailingEps', 'N/A'):.2f}" if isinstance(company_info.get('trailingEps'), (int, float)) else 'N/A',
                f"{company_info.get('dividendYield', 0)*100:.2f}%" if company_info.get('dividendYield') else 'N/A',
                f"{company_info.get('debtToEquity', 'N/A'):.2f}" if isinstance(company_info.get('debtToEquity'), (int, float)) else 'N/A',
                f"{company_info.get('returnOnEquity', 0)*100:.2f}%" if company_info.get('returnOnEquity') else 'N/A'
            ]
        }
        ratios_df = pd.DataFrame(ratios_data)
        st.dataframe(ratios_df, use_container_width=True)
    
    with tab3:
        # News and sentiment
        with st.spinner("Fetching latest news..."):
            news_data = fetch_news(ticker_symbol)
        
        if news_data:
            st.markdown("### Recent News")
            
            # Overall sentiment calculation
            all_sentiments = []
            
            for article in news_data:
                title = article.get('title', 'No title')
                publisher = article.get('publisher', 'Unknown')
                publish_time = article.get('providerPublishTime', 0)
                
                # Convert timestamp to readable date
                try:
                    publish_date = datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                except:
                    publish_date = 'Unknown date'
                
                # Sentiment analysis
                sentiment, score = analyze_sentiment(title)
                all_sentiments.append(score)
                
                # Determine CSS class for sentiment
                sentiment_class = {
                    'Positive': 'sentiment-positive',
                    'Negative': 'sentiment-negative',
                    'Neutral': 'sentiment-neutral'
                }.get(sentiment, 'sentiment-neutral')
                
                # Display news item
                st.markdown(f"""
                <div class="news-item {sentiment_class}">
                    <strong>{title}</strong><br>
                    <small>{publisher} | {publish_date}</small><br>
                    <span style="color: {'#10B981' if sentiment == 'Positive' else '#EF4444' if sentiment == 'Negative' else '#6B7280'};">
                        {sentiment} ({score:.2f})
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Overall sentiment gauge
            if all_sentiments:
                overall_sentiment = np.mean(all_sentiments)
                
                st.markdown("### Overall Sentiment")
                
                # Create gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = overall_sentiment,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [-1, -0.1], 'color': "#FEE2E2"},
                            {'range': [-0.1, 0.1], 'color': "#F3F4F6"},
                            {'range': [0.1, 1], 'color': "#D1FAE5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Interpretation
                if overall_sentiment > 0.1:
                    sentiment_text = "🟢 Positive"
                elif overall_sentiment < -0.1:
                    sentiment_text = "🔴 Negative"
                else:
                    sentiment_text = "🟡 Neutral"
                
                st.markdown(f"**Overall Market Sentiment: {sentiment_text}**")
        else:
            st.write("No recent news available for this ticker.")

else:
    st.info("👈 Please enter a stock ticker in the sidebar to get started.")