import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Custom CSS for background and buttons ---
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #1a73e8; color: white;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📊 Data Input")
data_source = st.sidebar.radio("Choose data source:", ("Upload Kragle Dataset", "Fetch Yahoo Finance Data"))

if data_source == "Upload Kragle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = None
else:
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)")
    fetch_btn = st.sidebar.button("Fetch Data")
    if fetch_btn and ticker:
        df = yf.download(ticker, period="1y")
    else:
        df = None

# --- Welcome Interface ---
st.title("💸 Financial ML App")
st.markdown("#### Welcome to your interactive finance ML dashboard!")
st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", width=300)
st.markdown("**Start by uploading a dataset or fetching stock data from Yahoo Finance.**")

# --- Step-by-Step ML Pipeline ---
if df is not None and not df.empty:
    st.success("Data loaded successfully!")
    if st.button("1️⃣ Preview Data"):
        st.dataframe(df.head())
        st.info("Here is a preview of your data.")

    if st.button("2️⃣ Preprocess Data"):
        st.write("Missing values before:", df.isnull().sum().sum())
        df = df.dropna()
        st.success("Missing values removed.")

    if st.button("3️⃣ Feature Engineering"):
        st.write("Feature engineering step (customize as needed).")
        # Example: st.write(df.describe())

    if st.button("4️⃣ Train/Test Split"):
        st.write("Splitting data...")
        # For demonstration, assume 'Close' is the target if present
        if 'Close' in df.columns:
            X = df.drop('Close', axis=1).select_dtypes(include=[np.number])
            y = df['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.success("Data split into train and test sets.")
            fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)])
            st.plotly_chart(fig)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
        else:
            st.warning("No 'Close' column found for regression. Please check your data.")

    if st.button("5️⃣ Train Model"):
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            model = LinearRegression()
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.session_state['model'] = model
            st.success("Linear Regression model trained!")
        else:
            st.warning("Please split the data first.")

    if st.button("6️⃣ Evaluate Model"):
        if 'model' in st.session_state and 'X_test' in st.session_state and 'y_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            score = st.session_state['model'].score(st.session_state['X_test'], st.session_state['y_test'])
            st.write(f"R2 Score: {score:.4f}")
            fig = px.scatter(x=st.session_state['y_test'], y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title='Actual vs Predicted')
            st.plotly_chart(fig)
        else:
            st.warning("Please train the model first.")

    if st.button("7️⃣ Visualize Results"):
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            st.dataframe(result_df.head())
            fig = px.line(result_df, title='Actual vs Predicted (Line Chart)')
            st.plotly_chart(fig)
        else:
            st.warning("Please train and evaluate the model first.")

    # Bonus: Download results
    if st.button("⬇️ Download Results"):
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv")
        else:
            st.warning("No results to download.")
else:
    st.warning("Please upload a dataset or fetch data to begin.")

# --- Themed GIF at End ---
st.markdown("---")
st.image("https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif", width=200)
st.markdown("**Thank you for using the Financial ML App!**") 
