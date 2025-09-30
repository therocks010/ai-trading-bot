# app.py - Streamlit Web Application (Ready for Cloud Deployment)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🤖 AI Trading Bot Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #00ff00;
        font-weight: bold;
    }
    .status-stopped {
        color: #ff4444;
        font-weight: bold;
    }
    .profit {
        color: #00ff00;
    }
    .loss {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

class SimpleTradingBot:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.positions = {}
        self.trades_history = []
        self.initial_capital = 10000
        self.current_capital = 10000
        
    def get_stock_data(self, symbol, period='6mo'):
        """Get stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        df = self.calculate_indicators(data)
        
        # Create target variable (next day return)
        df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = np.where(df['Future_Return'] > 0.01, 1,
                               np.where(df['Future_Return'] < -0.01, -1, 0))
        
        # Feature columns
        feature_columns = [
            'SMA_20', 'SMA_50', 'EMA_12', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'Volume_SMA', 'Volatility'
        ]
        
        # Normalize features
        for col in ['SMA_20', 'SMA_50', 'EMA_12', 'BB_Upper', 'BB_Lower']:
            if col in df.columns:
                df[col] = df[col] / df['Close']
        
        df.dropna(inplace=True)
        return df, feature_columns
    
    def train_model(self, symbol):
        """Train the ML model"""
        data = self.get_stock_data(symbol, period='1y')
        if data is None or len(data) < 100:
            return False
            
        df, feature_columns = self.prepare_features(data)
        
        if len(df) < 50:
            return False
        
        X = df[feature_columns]
        y = df['Target']
        
        # Remove hold signals for training
        mask = y != 0
        X = X[mask]
        y = y[mask]
        
        if len(X) < 20:
            return False
        
        # Train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.feature_columns = feature_columns
        self.is_trained = True
        return True
    
    def generate_signal(self, symbol):
        """Generate trading signal"""
        if not self.is_trained:
            return 0, 0
            
        data = self.get_stock_data(symbol, period='3mo')
        if data is None:
            return 0, 0
            
        df, _ = self.prepare_features(data)
        
        if len(df) < 10:
            return 0, 0
        
        # Get latest features
        latest_features = df[self.feature_columns].iloc[-1:].values
        
        if np.isnan(latest_features).any():
            return 0, 0
        
        # Generate prediction
        features_scaled = self.scaler.transform(latest_features)
        signal = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        # Apply trend analysis
        latest = df.iloc[-1]
        trend_score = sum([
            latest['Close'] > latest['SMA_20'],
            latest['Close'] > latest['SMA_50'],
            latest['SMA_20'] > latest['SMA_50'],
            30 < latest['RSI'] < 70,
            latest['MACD'] > latest['MACD_Signal']
        ])
        
        # Adaptive signal logic
        if trend_score >= 4 and signal == 1 and confidence > 0.6:
            return 1, confidence  # Strong buy
        elif trend_score <= 1 and signal == -1 and confidence > 0.6:
            return -1, confidence  # Strong sell
        else:
            return 0, confidence  # Hold
    
    def simulate_trade(self, symbol, signal, confidence):
        """Simulate a trade"""
        data = self.get_stock_data(symbol, period='1d')
        if data is None:
            return
            
        current_price = data['Close'].iloc[-1]
        position_size = int((self.current_capital * 0.1) / current_price)  # 10% of capital
        
        if signal == 1 and position_size > 0:  # Buy
            cost = position_size * current_price
            if cost <= self.current_capital:
                self.positions[symbol] = {
                    'type': 'long',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now()
                }
                self.current_capital -= cost
                
                trade = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': position_size,
                    'price': current_price,
                    'time': datetime.now(),
                    'confidence': confidence
                }
                self.trades_history.append(trade)
                
        elif signal == -1 and symbol in self.positions:  # Sell
            position = self.positions[symbol]
            if position['type'] == 'long':
                revenue = position['quantity'] * current_price
                self.current_capital += revenue
                
                pnl = revenue - (position['quantity'] * position['entry_price'])
                
                trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': position['quantity'],
                    'price': current_price,
                    'time': datetime.now(),
                    'pnl': pnl,
                    'confidence': confidence
                }
                self.trades_history.append(trade)
                del self.positions[symbol]

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = SimpleTradingBot()
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = ['AAPL', 'TSLA', 'SPY']

# Main header
st.markdown('<h1 class="main-header">🤖 AI Trading Bot Pro</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# API Configuration (Demo mode)
st.sidebar.warning("🔒 Demo Mode - No real trading")
st.sidebar.info("This is a simulation using real market data")

# Symbol selection
st.sidebar.subheader("📊 Stocks to Trade")
available_symbols = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'SPY', 'QQQ', 'NVDA', 'META', 'NFLX']
selected_symbols = st.sidebar.multiselect(
    "Select symbols:",
    available_symbols,
    default=st.session_state.selected_symbols
)
st.session_state.selected_symbols = selected_symbols

# Trading parameters
st.sidebar.subheader("🎯 Trading Parameters")
max_position = st.sidebar.slider("Max Position Size (%)", 5, 20, 10)
stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 5, 2)
take_profit = st.sidebar.slider("Take Profit (%)", 3, 10, 6)

# Control buttons
st.sidebar.subheader("🚀 Bot Control")

if st.sidebar.button("🚀 Start Trading Bot", type="primary"):
    if selected_symbols:
        st.session_state.trading_active = True
        st.sidebar.success("✅ Bot started!")
        
        # Train models for selected symbols
        progress_bar = st.sidebar.progress(0)
        for i, symbol in enumerate(selected_symbols):
            st.sidebar.write(f"Training model for {symbol}...")
            st.session_state.bot.train_model(symbol)
            progress_bar.progress((i + 1) / len(selected_symbols))
        
        st.sidebar.success("🧠 All models trained!")
    else:
        st.sidebar.error("❌ Please select at least one symbol")

if st.sidebar.button("⏹️ Stop Trading Bot"):
    st.session_state.trading_active = False
    st.sidebar.warning("⏹️ Bot stopped!")

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Account Balance",
        f"${st.session_state.bot.current_capital:,.2f}",
        f"{((st.session_state.bot.current_capital - st.session_state.bot.initial_capital) / st.session_state.bot.initial_capital * 100):+.2f}%"
    )

with col2:
    st.metric(
        "📈 Total Return",
        f"{((st.session_state.bot.current_capital - st.session_state.bot.initial_capital) / st.session_state.bot.initial_capital * 100):+.2f}%",
        f"${st.session_state.bot.current_capital - st.session_state.bot.initial_capital:+,.2f}"
    )

with col3:
    st.metric(
        "🔄 Active Positions",
        len(st.session_state.bot.positions),
        f"{len(st.session_state.bot.trades_history)} total trades"
    )

with col4:
    status_text = "🟢 RUNNING" if st.session_state.trading_active else "🔴 STOPPED"
    st.metric("🤖 Bot Status", status_text, "Live Demo Mode")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Signals", "💼 Positions", "📋 Trade History", "📈 Performance"])

with tab1:
    st.header("📊 Live Trading Signals")
    
    if st.session_state.trading_active and selected_symbols:
        if st.button("🔄 Generate New Signals"):
            signals_data = []
            
            for symbol in selected_symbols:
                with st.spinner(f"Analyzing {symbol}..."):
                    signal, confidence = st.session_state.bot.generate_signal(symbol)
                    
                    # Get current price
                    data = st.session_state.bot.get_stock_data(symbol, period='1d')
                    current_price = data['Close'].iloc[-1] if data is not None else 0
                    
                    signal_text = "🟢 BUY" if signal == 1 else "🔴 SELL" if signal == -1 else "⚪ HOLD"
                    
                    signals_data.append({
                        'Symbol': symbol,
                        'Signal': signal_text,
                        'Confidence': f"{confidence:.2%}",
                        'Current Price': f"${current_price:.2f}",
                        'Action': 'Execute Trade' if signal != 0 else 'Monitor'
                    })
                    
                    # Simulate trade if signal is strong
                    if signal != 0 and confidence > 0.6:
                        st.session_state.bot.simulate_trade(symbol, signal, confidence)
            
            # Display signals table
            if signals_data:
                df_signals = pd.DataFrame(signals_data)
                st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("🚀 Start the trading bot to see live signals!")

with tab2:
    st.header("💼 Current Positions")
    
    if st.session_state.bot.positions:
        positions_data = []
        total_pnl = 0
        
        for symbol, position in st.session_state.bot.positions.items():
            # Get current price
            data = st.session_state.bot.get_stock_data(symbol, period='1d')
            current_price = data['Close'].iloc[-1] if data is not None else position['entry_price']
            
            pnl = (current_price - position['entry_price']) * position['quantity']
            pnl_pct = (pnl / (position['entry_price'] * position['quantity'])) * 100
            total_pnl += pnl
            
            positions_data.append({
                'Symbol': symbol,
                'Type': position['type'].upper(),
                'Quantity': position['quantity'],
                'Entry Price': f"${position['entry_price']:.2f}",
                'Current Price': f"${current_price:.2f}",
                'P&L': f"${pnl:+.2f}",
                'P&L %': f"{pnl_pct:+.2f}%",
                'Entry Time': position['entry_time'].strftime("%Y-%m-%d %H:%M")
            })
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True)
        
        # Total P&L
        st.metric("💵 Total Unrealized P&L", f"${total_pnl:+.2f}")
    else:
        st.info("📭 No active positions")

with tab3:
    st.header("📋 Trade History")
    
    if st.session_state.bot.trades_history:
        trades_data = []
        
        for trade in st.session_state.bot.trades_history[-20:]:  # Last 20 trades
            trades_data.append({
                'Time': trade['time'].strftime("%Y-%m-%d %H:%M:%S"),
                'Symbol': trade['symbol'],
                'Action': trade['action'],
                'Quantity': trade['quantity'],
                'Price': f"${trade['price']:.2f}",
                'Confidence': f"{trade['confidence']:.2%}",
                'P&L': f"${trade.get('pnl', 0):+.2f}" if 'pnl' in trade else 'N/A'
            })
        
        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("📭 No trades executed yet")

with tab4:
    st.header("📈 Performance Analysis")
    
    # Create sample performance chart
    if st.session_state.bot.trades_history:
        # Calculate cumulative returns
        dates = [trade['time'] for trade in st.session_state.bot.trades_history]
        returns = [st.session_state.bot.current_capital]
        
        # Simple performance visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=returns,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00ff41', width=3)
        ))
        
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_trades = len(st.session_state.bot.trades_history)
        winning_trades = len([t for t in st.session_state.bot.trades_history if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%", f"{winning_trades}/{total_trades}")
    
    with col2:
        total_return = ((st.session_state.bot.current_capital - st.session_state.bot.initial_capital) / st.session_state.bot.initial_capital * 100)
        st.metric("📊 Total Return", f"{total_return:+.2f}%")
    
    with col3:
        st.metric("🔄 Total Trades", total_trades)

# Real-time updates
if st.session_state.trading_active:
    # Auto-refresh every 30 seconds
    time.sleep(0.1)  # Small delay to prevent too frequent updates
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 AI Trading Bot Pro - Demo Version | Built with Streamlit</p>
    <p>⚠️ This is a simulation for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
