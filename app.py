# app.py
import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="ShopSmart Premium Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Premium CSS Styling (Fixed parameter here)
st.markdown("""
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Header Card */
    .header-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
    }
    .header-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
    }
    
    /* Section Cards */
    .section-card {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 1.5rem;
    }
    
    /* Result Banners */
    .result-card-buy {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #10b981;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .result-card-browse {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #f59e0b;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    }
    </style>
""", unsafe_allow_html=True)  # <-- FIXED

# 3. Load ML Artifacts securely
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    model = joblib.load("artifacts/model.pkl")
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error("⚠️ Artifacts not found. Please run 'python src/train.py' first to generate model.pkl and preprocessor.pkl.")
    st.stop()

# 4. Premium Header App Banner (Fixed parameter here)
st.markdown("""
    <div class="header-box">
        <div class="header-title">🛒 ShopSmart AI Analytics</div>
        <div class="header-subtitle">Predicting e-commerce intent and converting user sessions into revenue insights using Machine Learning.</div>
    </div>
""", unsafe_allow_html=True)  # <-- FIXED

# 5. UI Layout - Interactive Input Panel split into columns
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="section-card"><h3>🕒 Session Activity & Durations</h3></div>', unsafe_allow_html=True)  # <-- FIXED
    c1, c2, c3 = st.columns(3)
    with c1:
        admin = st.number_input("🔗 Administrative Pages", min_value=0, value=1, help="Pages viewed related to account management.")
        admin_dur = st.number_input("⏱️ Admin Duration (sec)", min_value=0.0, value=15.0)
    with c2:
        info = st.number_input("ℹ️ Informational Pages", min_value=0, value=0, help="Pages like FAQs or policy guidelines.")
        info_dur = st.number_input("⏱️ Info Duration (sec)", min_value=0.0, value=0.0)
    with c3:
        prod = st.number_input("📦 Product Related Pages", min_value=0, value=12, help="Specific item display pages.")
        prod_dur = st.number_input("⏱️ Product Duration (sec)", min_value=0.0, value=240.0)

    st.markdown('<div class="section-card"><h3>📉 Engagement & Exit Metrics</h3></div>', unsafe_allow_html=True)  # <-- FIXED
    c4, c5, c6 = st.columns(3)
    with c4:
        bounce = st.slider("🚨 Bounce Rate", 0.0, 1.0, 0.02, step=0.005, help="Percentage of users leaving immediately from this page entry point.")
    with c5:
        exit_rate = st.slider("🚪 Exit Rate", 0.0, 1.0, 0.04, step=0.005, help="Percentage of overall session exits happening from this page view.")
    with c6:
        page_val = st.number_input("💎 Page Value Score", min_value=0.0, value=18.5, help="Average value of this page relative to previous successful targets.")

with col_right:
    st.markdown('<div class="section-card"><h3>🌍 Visitor Context</h3></div>', unsafe_allow_html=True)  # <-- FIXED
    month = st.selectbox("📅 Month of Session", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
    special_day = st.slider("🎁 Closeness to Special Day", 0.0, 1.0, 0.0, help="Proximity of session to major holidays/events.")
    visitor = st.selectbox("👤 Visitor Category", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("🥂 Weekend Transaction", value=False)
    
    # Advanced / Encoded System Elements collapsed to keep UI clean
    with st.expander("🛠️ Advanced Technical Metas"):
        os_sys = st.slider("Operating System ID", 1, 8, 2)
        browser = st.slider("Browser Engine ID", 1, 13, 2)
        region = st.slider("Geographic Region ID", 1, 9, 1)
        traffic = st.slider("Traffic Routing Channel", 1, 20, 2)

# 6. Prediction Engine Trigger Execution
st.markdown("---")
if st.button("🚀 Run Intent Inference Engine", use_container_width=True):
    # Assemble structured Dataframe matching pipeline inputs
    input_df = pd.DataFrame([{
        'Administrative': admin, 'Administrative_Duration': admin_dur,
        'Informational': info, 'Informational_Duration': info_dur,
        'ProductRelated': prod, 'ProductRelated_Duration': prod_dur,
        'BounceRates': bounce, 'ExitRates': exit_rate, 'PageValues': page_val,
        'SpecialDay': special_day, 'Month': month, 'OperatingSystems': os_sys,
        'Browser': browser, 'Region': region, 'TrafficType': traffic,
        'VisitorType': visitor, 'Weekend': weekend
    }])
    
    # Process and Predict
    transformed_features = preprocessor.transform(input_df)
    prediction = model.predict(transformed_features)[0]
    probabilities = model.predict_proba(transformed_features)[0]
    
    # Aesthetic Premium Output Cards (Fixed parameters here)
    if prediction == 1:
        st.markdown(f"""
            <div class="result-card-buy">
                <h2 style="color: #10b981; margin: 0;">💰 High Conversion Intent Detected!</h2>
                <p style="color: #a7f3d0; font-size: 1.2rem; margin-top: 0.5rem;">
                    The model predicts this visitor is highly likely to make a purchase.<br>
                    <b>Confidence Score: {probabilities[1]*100:.1f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)  # <-- FIXED
    else:
        st.markdown(f"""
            <div class="result-card-browse">
                <h2 style="color: #f59e0b; margin: 0;">🚶 Casual Window Shopping Behavior</h2>
                <p style="color: #fde68a; font-size: 1.2rem; margin-top: 0.5rem;">
                    The model predicts this session will end without a transaction.<br>
                    <b>Confidence Score: {probabilities[0]*100:.1f}%</b>
                </p>
            </div>
        """, unsafe_allow_html=True)  # <-- FIXED
