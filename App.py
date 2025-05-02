import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Custom CSS for background, buttons, and vertical layout ---
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: 1px solid #1a73e8;
        font-weight: bold;
        transition: 0.2s;
        display: block;
        width: 100%;
        margin-bottom: 0.5em;
        padding: 0.5em;
    }
    .stButton>button:hover {
        background-color: #1761b0;
        color: #FFD700;
        border: 1px solid #FFD700;
    }
    .stSidebar {
        background-color: #f0f2f6 !important;
    }
    .main {background-color: #f0f2f6;}
    h1, h2, h3 {color: #1a73e8;}
    .stAlert {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://images.pexels.com/photos/1181696/pexels-photo-1181696.jpeg?auto=compress&w=256&h=256&fit=facearea", width=80)
st.sidebar.title("📊 Data Input")
data_source = st.sidebar.radio("Choose data source:", ("Upload Kaggle Dataset", "Fetch Yahoo Finance Data"))

if data_source == "Upload Kaggle Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            df = None
    else:
        df = None
else:
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL)", value="AAPL")
    if "yahoo_df" not in st.session_state:
        st.session_state["yahoo_df"] = None
    if st.sidebar.button("Fetch Data") and ticker:
        try:
            st.session_state["yahoo_df"] = yf.download(ticker, period="1y")
            if st.session_state["yahoo_df"].empty:
                st.error("No data found for the given ticker.")
                st.session_state["yahoo_df"] = None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.session_state["yahoo_df"] = None
    df = st.session_state["yahoo_df"]

# --- Welcome Interface ---
st.title("💸 Financial ML App")
st.markdown("#### Welcome to your interactive finance ML dashboard!")
st.image("https://media.giphy.com/media/l0Iyl55kTeh71nTWw/giphy.gif", width=300)
st.markdown("**Start by uploading a dataset or fetching stock data from Yahoo Finance.**")

# --- Step-by-Step ML Pipeline ---
if df is not None and not df.empty:
    st.success("Data loaded successfully!")
    
    # Vertical pipeline buttons
    st.header("Machine Learning Pipeline")
    
    if st.button("1️⃣ Preview Data"):
        st.dataframe(df.head())
        st.info("Here is a preview of your data.")

    if st.button("2️⃣ Preprocess Data"):
        missing_count = df.isnull().sum().sum()
        st.write(f"Missing values before: {missing_count}")
        df = df.dropna()
        st.success("Missing values removed.")
        st.write("Data after preprocessing:")
        st.dataframe(df.head())

    if st.button("3️⃣ Feature Engineering"):
        st.write("Feature engineering step (customized for stock data).")
        if 'Close' in df.columns:
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['Returns'] = df['Close'].pct_change()
            df = df.dropna()
            st.session_state['engineered_df'] = df
            st.success("Features added: 10-day MA, 50-day MA, Returns.")
            st.write("Engineered features:")
            st.dataframe(df[['Close', 'MA10', 'MA50', 'Returns']].head())
        else:
            st.warning("No 'Close' column found for feature engineering.")

    if st.button("4️⃣ Train/Test Split"):
        if 'engineered_df' in st.session_state:
            df = st.session_state['engineered_df']
            if 'Close' in df.columns:
                X = df[['MA10', 'MA50', 'Returns']].select_dtypes(include=[np.number])
                y = df['Close']
                if X.empty or y.empty:
                    st.error("No valid features for training. Check your data.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.success("Data split into train and test sets.")
                    fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)], title="Train/Test Split")
                    st.plotly_chart(fig)
            else:
                st.warning("No 'Close' column found for regression.")
        else:
            st.warning("Please perform feature engineering first.")

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
            st.write(f"R² Score: {score:.4f}")
            fig = px.scatter(x=st.session_state['y_test'], y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title='Actual vs Predicted')
            st.plotly_chart(fig)
            st.session_state['y_pred'] = y_pred
            st.success("Model evaluated successfully!")
        else:
            st.warning("Please train the model first.")

    if st.button("7️⃣ Visualize Results"):
        if 'y_pred' in st.session_state and 'y_test' in st.session_state:
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']})
            st.dataframe(result_df.head())
            fig = px.line(result_df, title='Actual vs Predicted (Line Chart)')
            st.plotly_chart(fig)
            st.success("Results visualized successfully!")
        else:
            st.warning("Please evaluate the model first.")

    # Bonus: Download results
    if st.button("⬇️ Download Results"):
        if 'y_pred' in st.session_state and 'y_test' in st.session_state:
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': st.session_state['y_pred']})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv", mime="text/csv")
            st.success("Results ready for download!")
        else:
            st.warning("No results to download.")
else:
    st.warning("Please upload a dataset or fetch data to begin.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by Sumbal Murtaza for AF3005 – Programming for Finance, Spring 2025")
