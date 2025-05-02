import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import time

# --- Custom CSS for a clean, light look and attractive buttons ---
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
st.sidebar.title("📊 Data Input")
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
                    st.sidebar.error("No data found for this ticker. Please check the symbol and try again.")
        else:
            st.sidebar.warning("Please enter a ticker symbol.")
    df = st.session_state["yahoo_df"]

# --- Welcome Interface ---
st.title("💸 Financial ML App")
st.markdown("#### Welcome to your interactive finance ML dashboard!")
st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", width=300)
st.markdown("**Start by uploading a dataset or fetching stock data from Yahoo Finance.**")

# --- Step-by-Step ML Pipeline ---
if df is not None and not df.empty:
    st.success("Data loaded successfully!")
    if st.button("1️⃣ Preview Data", help="Show the first few rows of your data."):
        st.dataframe(df.head())
        st.info("Here is a preview of your data.")

    if st.button("2️⃣ Preprocess Data", help="Remove missing values and clean your data."):
        progress = st.progress(0, text="Preprocessing data...")
        st.write("Missing values before:", df.isnull().sum().sum())
        for percent in range(0, 101, 20):
            time.sleep(0.1)
            progress.progress(percent, text=f"Preprocessing... {percent}%")
        df = df.dropna()
        progress.progress(100, text="Preprocessing complete!")
        st.success("Missing values removed.")

    if st.button("3️⃣ Feature Engineering", help="View summary statistics and features."):
        progress = st.progress(0, text="Engineering features...")
        for percent in range(0, 101, 25):
            time.sleep(0.08)
            progress.progress(percent, text=f"Engineering features... {percent}%")
        st.write("Feature engineering step (customize as needed).")
        st.write(df.describe())
        progress.progress(100, text="Feature engineering complete!")

    if st.button("4️⃣ Train/Test Split", help="Split your data into training and testing sets."):
        st.write("Splitting data...")
        if 'Close' in df.columns:
            X = df.drop('Close', axis=1).select_dtypes(include=[np.number])
            y = df['Close']
            if X.shape[1] == 0:
                st.warning("No numeric features available for training. Please check your data.")
            else:
                progress = st.progress(0, text="Splitting data...")
                for percent in range(0, 101, 33):
                    time.sleep(0.07)
                    progress.progress(percent, text=f"Splitting... {percent}%")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                st.success("Data split into train and test sets.")
                fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)])
                st.plotly_chart(fig)
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                progress.progress(100, text="Split complete!")
        else:
            st.warning("No 'Close' column found for regression. Please check your data.")

    if st.button("5️⃣ Train Model", help="Train a Linear Regression model on your data."):
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            progress = st.progress(0, text="Training model...")
            for percent in range(0, 101, 25):
                time.sleep(0.12)
                progress.progress(percent, text=f"Training... {percent}%")
            model = LinearRegression()
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.session_state['model'] = model
            progress.progress(100, text="Model training complete!")
            st.success("Linear Regression model trained!")
            st.balloons()
        else:
            st.warning("Please split the data first.")

    if st.button("6️⃣ Evaluate Model", help="Evaluate the trained model's performance."):
        if 'model' in st.session_state and 'X_test' in st.session_state and 'y_test' in st.session_state:
            progress = st.progress(0, text="Evaluating model...")
            for percent in range(0, 101, 20):
                time.sleep(0.09)
                progress.progress(percent, text=f"Evaluating... {percent}%")
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            score = st.session_state['model'].score(st.session_state['X_test'], st.session_state['y_test'])
            st.write(f"R2 Score: {score:.4f}")
            fig = px.scatter(x=st.session_state['y_test'], y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title='Actual vs Predicted')
            st.plotly_chart(fig)
            progress.progress(100, text="Evaluation complete!")
            st.snow()
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

# --- Themed GIF at End ---
st.markdown("---")
st.image("https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif", width=200)
st.markdown("**Thank you for using the Financial ML App!**") 
