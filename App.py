import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import time

# --- Custom CSS for sidebar, cards, buttons, and hover effects ---
st.markdown("""
    <style>
    .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2 {
        background-color: #f7fafd !important;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.08);
    }
    .stSidebar {
        background: linear-gradient(135deg, #e3f0ff 0%, #f7fafd 100%) !important;
        box-shadow: 2px 0 8px rgba(30, 64, 175, 0.10);
    }
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
        box-shadow: 0 2px 8px #1a73e8;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: bold;
        color: #1a73e8;
    }
    .stTabs [aria-selected="true"] {
        background: #e3f0ff;
        border-radius: 8px 8px 0 0;
    }
    .custom-card {
        background: #e3f0ff;
        border-radius: 16px;
        padding: 1.5em;
        box-shadow: 0 2px 12px rgba(30, 64, 175, 0.10);
        margin-bottom: 1.5em;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #e3f0ff;
        color: #1a73e8;
        text-align: center;
        padding: 0.5em 0;
        font-size: 16px;
        border-top: 1px solid #b3d1f7;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Logo and Info ---
st.sidebar.image("https://images.pexels.com/photos/1181696/pexels-photo-1181696.jpeg?auto=compress&w=256&h=256&fit=facearea", width=80)  # Working Pexels lady in finance image
st.sidebar.title("📊 Financial ML App")
st.sidebar.markdown("""
**Welcome!**
- Upload your own CSV (from Kaggle or elsewhere)
- Or fetch real-time stock data from Yahoo Finance (e.g., AAPL)
- If Yahoo fetch fails, download from [Yahoo Finance](https://finance.yahoo.com/) or [Kaggle](https://www.kaggle.com/datasets?search=stock+prices) and upload here.
""")

# --- Data Source Selection ---
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

# --- Main App Title and Welcome ---
st.image("https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", width=350)  # Working Giphy finance GIF
st.markdown("""
# 💸 <span style='color:#1a73e8'>Financial ML Dashboard</span>
""", unsafe_allow_html=True)
st.markdown("""
<div class='custom-card'>
    <span style='font-size:22px;color:#1a73e8'><b>Welcome to your interactive finance ML dashboard!</b></span><br>
    <span style='font-size:16px;'>Start by uploading a dataset or fetching stock data from Yahoo Finance.<br>
    Step through the workflow tabs below to build and evaluate your model.</span>
</div>
""", unsafe_allow_html=True)

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

# --- Step-by-Step ML Pipeline in Tabs ---
if df is not None and not df.empty:
    st.success("Data loaded successfully!")
    tabs = st.tabs([
        "1️⃣ Preview Data", "2️⃣ Preprocess Data", "3️⃣ Feature Engineering", "4️⃣ Train/Test Split", "5️⃣ Train Model", "6️⃣ Evaluate Model", "7️⃣ Visualize Results", "⬇️ Download Results"
    ])

    with tabs[0]:
        st.markdown("### 1️⃣ Preview Data")
        st.dataframe(df.head())
        st.info("Here is a preview of your data.")

    with tabs[1]:
        st.markdown("### 2️⃣ Preprocess Data")
        st.write("Missing values before:", df.isnull().sum().sum())
        progress = st.progress(0, text="Preprocessing data...")
        for percent in range(0, 101, 20):
            time.sleep(0.1)
            progress.progress(percent, text=f"Preprocessing... {percent}%")
        df = df.dropna()
        progress.progress(100, text="Preprocessing complete!")
        st.success("Missing values removed.")

    with tabs[2]:
        st.markdown("### 3️⃣ Feature Engineering")
        progress = st.progress(0, text="Engineering features...")
        for percent in range(0, 101, 25):
            time.sleep(0.08)
            progress.progress(percent, text=f"Engineering features... {percent}%")
        st.write("Feature engineering step (customize as needed).")
        st.write(df.describe())
        progress.progress(100, text="Feature engineering complete!")

    with tabs[3]:
        st.markdown("### 4️⃣ Train/Test Split")
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
                progress = st.progress(0, text="Splitting data...")
                for percent in range(0, 101, 33):
                    time.sleep(0.07)
                    progress.progress(percent, text=f"Splitting... {percent}%")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                st.success("Data split into train and test sets.")
                fig = px.pie(names=["Train", "Test"], values=[len(X_train), len(X_test)], title="Train/Test Split")
                st.plotly_chart(fig)
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['target_col'] = target_col
                progress.progress(100, text="Split complete!")

    with tabs[4]:
        st.markdown("### 5️⃣ Train Model")
        if 'X_train' in st.session_state and 'y_train' in st.session_state:
            progress = st.progress(0, text="Training model...")
            for percent in range(0, 101, 25):
                time.sleep(0.12)
                progress.progress(percent, text=f"Training... {percent}%")
            model = LinearRegression()
            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.session_state['model'] = model
            progress.progress(100, text="Model training complete!")
            st.success("💹 Model trained! Ready to make financial predictions.")
        else:
            st.warning("Please split the data first.")

    with tabs[5]:
        st.markdown("### 6️⃣ Evaluate Model")
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
            st.success("Model evaluation complete! 🎉")
            st.image("https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", width=350)  # Working Giphy finance GIF
        else:
            st.warning("Please train the model first.")

    with tabs[6]:
        st.markdown("### 7️⃣ Visualize Results")
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            st.dataframe(result_df.head())
            fig = px.line(result_df, title='Actual vs Predicted (Line Chart)')
            st.plotly_chart(fig)
        else:
            st.warning("Please train and evaluate the model first.")

    with tabs[7]:
        st.markdown("### ⬇️ Download Results")
        if 'model' in st.session_state and 'X_test' in st.session_state:
            y_pred = st.session_state['model'].predict(st.session_state['X_test'])
            result_df = pd.DataFrame({'Actual': st.session_state['y_test'], 'Predicted': y_pred})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv")
        else:
            st.warning("No results to download.")
else:
    st.info("Please upload a dataset or fetch data to begin.")

# --- Custom Footer ---
st.markdown("""
<div class='footer'>
    <b>Developed by [Your Name] | AF3005 – Programming for Finance | <span style='font-size:20px;'>💸</span></b>
</div>
""", unsafe_allow_html=True) 
