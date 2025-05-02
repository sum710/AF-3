import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import time

# --- Custom CSS for attractive buttons ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: 1px solid #1a73e8;
        font-weight: bold;
        transition: 0.2s;
        margin-bottom: 0.5em;
    }
    .stButton>button:hover {
        background-color: #1761b0;
        color: #FFD700;
        border: 1px solid #FFD700;
    }
    .stSidebar {
        background-color: #f0f2f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?auto=format&fit=facearea&w=256&h=256&q=80", width=80)
st.sidebar.title("📊 Financial ML App")
data_source = st.sidebar.radio(
    "Choose data source:",
    ("Upload Kragle Dataset", "Fetch Yahoo Finance Data"),
    help="Select whether to upload your own CSV or fetch real-time stock data."
)

if data_source == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV", type=["csv"], help="Upload a CSV file containing your financial data."
    )
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Kragle dataset uploaded!")
    else:
        df = None
else:
    ticker = st.sidebar.text_input(
        "Enter Stock Ticker (e.g. AAPL)", help="Type a valid stock ticker symbol, e.g., AAPL for Apple."
    )
    if "yahoo_df" not in st.session_state:
        st.session_state["yahoo_df"] = None
    if st.sidebar.button("Fetch Data", help="Fetch 1 year of daily data for the entered ticker."):
        if ticker:
            with st.spinner(f"Fetching data for {ticker}..."):
                data = yf.download(ticker, period="1y")
                if data is not None and not data.empty:
                    st.session_state["yahoo_df"] = data
                    st.sidebar.success(f"Data for {ticker} loaded!")
                else:
                    st.session_state["yahoo_df"] = None
                    st.sidebar.error("No data found for this ticker. Please check the symbol and try again.\n\nTroubleshooting tips:\n- Make sure your internet connection is working.\n- Try a well-known ticker like AAPL.\n- If it still fails, update yfinance with 'pip install --upgrade yfinance'.\n- If you are behind a firewall or proxy, try a different network.\n- You can also manually download from [Yahoo Finance](https://finance.yahoo.com/) or [Kaggle](https://www.kaggle.com/datasets?search=stock+prices) and upload.")
        else:
            st.sidebar.warning("Please enter a ticker symbol.")
    df = st.session_state["yahoo_df"]

# --- Welcome Interface ---
st.title("💸 Financial ML App")
st.markdown("#### Welcome to your interactive finance ML dashboard!")
st.image("https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif", width=300)
st.markdown("**Start by uploading a dataset or fetching stock data from Yahoo Finance.**")

# --- Quick Data Summary Charts (if data loaded and target selected) ---
if 'target_col' in st.session_state and df is not None and not df.empty:
    st.markdown("## 📊 Quick Data Overview")
    col1, col2 = st.columns(2)
    target_col = st.session_state['target_col']
    with col1:
        st.markdown(f"#### {target_col} Over Time")
        if 'Date' in df.columns:
            chart_df = df.copy()
            chart_df['Date'] = pd.to_datetime(chart_df['Date'])
            chart_df = chart_df.sort_values('Date')
            st.line_chart(chart_df.set_index('Date')[target_col])
        else:
            st.line_chart(df[target_col])
    with col2:
        st.markdown(f"#### {target_col} Distribution")
        st.bar_chart(df[target_col].value_counts().sort_index() if df[target_col].nunique() < 30 else df[target_col])

# --- Step-by-Step ML Pipeline ---
if df is not None and not df.empty:
    st.success("Data loaded successfully!")
    if st.button("1️⃣ Preview Data", help="Show the first few rows of your data."):
        st.dataframe(df.head())
        st.info("Here is a preview of your data.")

    if st.button("2️⃣ Preprocess Data", help="Remove missing values and clean your data."):
        st.write("Missing values before:", df.isnull().sum().sum())
        df = df.dropna()
        st.success("Missing values removed.")

    if st.button("3️⃣ Feature Engineering", help="View summary statistics and features."):
        st.write("Feature engineering step (customize as needed).")
        st.write(df.describe())

    if st.button("4️⃣ Train/Test Split", help="Split your data into training and testing sets."):
        st.write("Splitting data...")
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found for regression. Please check your data.")
        else:
            close_candidates = [col for col in numeric_cols if col.lower() == 'close']
            default_col = close_candidates[0] if close_candidates else numeric_cols[0]
            target_col = st.selectbox(
                "Select the target column for regression:",
                options=numeric_cols,
                index=numeric_cols.index(default_col),
                help="Choose the column you want to predict (e.g., Close, Adj Close, etc.)"
            )
            X = df.drop(target_col, axis=1).select_dtypes(include=[np.number])
            y = df[target_col]
            if X.shape[1] == 0:
                st.warning("No numeric features available for training. Please check your data.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                st.success("Data split into train and test sets.")
                fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)])
                st.plotly_chart(fig)
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['target_col'] = target_col

    if st.button("5️⃣ Train Model", help="Train a Linear Regression model on your data."):
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            model = LinearRegression()
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.session_state['model'] = model
            st.success("💹 Model trained! Ready to make financial predictions.")
        else:
            st.warning("Please split the data first.")

    if st.button("6️⃣ Evaluate Model", help="Evaluate the trained model's performance."):
        if 'model' in st.session_state and 'X_test' in st.session_state and 'y_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            score = st.session_state['model'].score(st.session_state['X_test'], st.session_state['y_test'])
            st.write(f"R2 Score: {score:.4f}")
            fig = px.scatter(x=st.session_state['y_test'], y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title='Actual vs Predicted')
            st.plotly_chart(fig)
            st.success("Model evaluation complete! 🎉")
            st.image("https://media.giphy.com/media/13HgwGsXF0aiGY/giphy.gif", width=300)
        else:
            st.warning("Please train the model first.")

    if st.button("7️⃣ Visualize Results", help="Visualize predictions vs. actual values."):
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            st.dataframe(result_df.head())
            fig = px.line(result_df, title='Actual vs Predicted (Line Chart)')
            st.plotly_chart(fig)
        else:
            st.warning("Please train and evaluate the model first.")

    if st.button("⬇️ Download Results", help="Download the predictions as a CSV file."):
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv")
        else:
            st.warning("No results to download.")
else:
    st.info("Please upload a dataset or fetch data to begin.") 
