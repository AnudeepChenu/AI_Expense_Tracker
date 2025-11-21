import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from sklearn.linear_model import LinearRegression

# ----------------------------
# CONFIGURATION
# ----------------------------
API_KEY = "AIzaSyBNp701g2GN_9RCYAq6nqXsWvFxvowk6W8"  # Replace with your actual key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
st.set_page_config(page_title="💰 AI Expense Analyzer (₹)", layout="wide")

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------

def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(i) for i in obj]
    else:
        return obj

def preprocess_data(df):
    """Handle columns and types for this dataset."""
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount (INR)'] = pd.to_numeric(df['Amount (INR)'])
    except Exception as e:
        st.error(f"⚠️ Error while parsing: {e}")
        return None

    df = df.dropna(subset=['Date', 'Amount (INR)'])
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df = df.sort_values(by='Date')
    return df

def get_ml_prediction(monthly_totals):
    """Simple regression-based spending forecast."""
    if len(monthly_totals) < 2:
        return "Not enough data to predict."

    X = np.arange(len(monthly_totals)).reshape(-1, 1)
    y = monthly_totals.values

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict([[len(monthly_totals)]])[0]
    return f"₹{pred:,.2f}"

def get_ai_summary(monthly_data, category_data):
    if API_KEY == "YOUR_GEMINI_API_KEY":
        st.error("Please add your Gemini API key.")
        return None

    prompt_data = {
        "monthly_spending": monthly_data.to_dict(),
        "category_spending": category_data.to_dict()
    }
    prompt_data = convert_keys_to_str(prompt_data)

    system_prompt = (
        "You are an Indian financial advisor. All amounts are in ₹ (Rupees). "
        "Analyze spending trends, top categories, and suggest one actionable tip to save money."
    )

    user_query = f"Here is my expense data:\n{json.dumps(prompt_data, indent=2)}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    try:
        response = requests.post(GEMINI_API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            return result['candidates'][0]['content']['parts'][0]['text']
        return "AI summary could not be generated."
    except Exception as e:
        return f"Error: {e}"

# ----------------------------
# MODERN STYLING
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #f8fafc;
    font-family: "Inter", sans-serif;
}
.card {
    background: white;
    padding: 1.2rem 1.5rem;
    border-radius: 15px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
}
[data-testid="stMetricValue"] {
    font-size: 2rem;
    color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# MAIN APP
# ----------------------------
def main():
    st.title("Expense Tracker & Analyzer")

    st.sidebar.header("📂 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your Expense CSV", type="csv")

    if st.sidebar.button("Use Sample Data"):
        st.session_state.df = pd.read_csv("sample_expenses.csv")
        st.sidebar.success("✅ Sample data loaded!")

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")

    if 'df' not in st.session_state:
        st.info("⬆️ Upload a CSV or use sample data to begin.")
        return

    df = preprocess_data(st.session_state.df)
    if df is None:
        return

    # --- Data Overview ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📋 Data Overview")
    st.dataframe(df[['Date', 'Category', 'Amount (INR)', 'Description']].head(20))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Spending Analysis ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Spending Analysis")

    col1, col2 = st.columns(2)
    with col1:
        monthly_spending = df.groupby('YearMonth')['Amount (INR)'].sum()
        monthly_spending.index = monthly_spending.index.to_timestamp()
        st.subheader("📅 Monthly Spending")
        st.line_chart(monthly_spending)

    with col2:
        category_spending = df.groupby('Category')['Amount (INR)'].sum().sort_values(ascending=False)
        st.subheader("🛍️ Spending by Category")
        st.bar_chart(category_spending)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ML Prediction ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🤖 Next Month’s Spending Prediction")
    pred = get_ml_prediction(monthly_spending)
    st.metric(label="Predicted Spend (Next Month)", value=pred)
    st.caption("Based on a simple linear regression model.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- AI Summary ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("💡 AI-Powered Insights")
    if st.button("✨ Generate AI Summary"):
        with st.spinner("Analyzing your expenses..."):
            monthly_summary = df.groupby('YearMonth')['Amount (INR)'].sum()
            category_summary = df.groupby('Category')['Amount (INR)'].sum()
            monthly_summary.index = monthly_summary.index.astype(str)

            summary = get_ai_summary(monthly_summary, category_summary)
            st.success("✅ Summary generated")
            st.markdown(summary)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
