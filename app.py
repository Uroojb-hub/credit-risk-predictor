# 1 Good (Lower Risk) 0 Bad (Higher Risk)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration for clean layout
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Slate / Minimal Palette
st.markdown("""
    <style>
    /* Global background */
    .stApp {
        background-color: #1E222A;
        color: #F3F4F6;
    }
    
    /* Card containers */
    .metric-card {
        background-color: #282C34;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #3E4451;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    
    /* Status typography */
    .status-good {
        color: #10B981;
        font-weight: 700;
        font-size: 22px;
        letter-spacing: 0.5px;
    }
    .status-bad {
        color: #EF4444;
        font-weight: 700;
        font-size: 22px;
        letter-spacing: 0.5px;
    }
    
    /* Custom button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        background-color: #3B82F6;
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border: none;
    }

    /* Headings and subtext */
    h1, h2, h3, h4, h5 {
        color: #FFFFFF !important;
    }
    p, span, label {
        color: #D1D5DB !important;
    }
    </style>
""", unsafe_allow_html=True)

# Set Dark Matplotlib & Seaborn Theme for charts
plt.style.use('dark_background')
sns.set_theme(style="ticks", rc={
    "axes.facecolor": "#282C34",
    "figure.facecolor": "#282C34",
    "text.color": "#F3F4F6",
    "axes.labelcolor": "#D1D5DB",
    "xtick.color": "#9CA3AF",
    "ytick.color": "#9CA3AF",
    "grid.color": "#3E4451"
})

# 2. Asset & Data Loading
@st.cache_resource
def load_models():
    model = joblib.load('extra_trees_credit_model.pkl')
    sex_enc = joblib.load('Sex_encoder.pkl')
    housing_enc = joblib.load('Housing_encoder.pkl')
    saving_enc = joblib.load('Saving accounts_encoder.pkl')
    checking_enc = joblib.load('Checking account_encoder.pkl')
    return model, sex_enc, housing_enc, saving_enc, checking_enc

@st.cache_data
def load_data():
    return pd.read_csv('german_credit_data.csv', index_col=0)

try:
    model, sex_enc, housing_enc, saving_enc, checking_enc = load_models()
    df = load_data()
except Exception as e:
    st.error(f"Error loading required model or data files: {e}")
    st.stop()

# 3. Application Header
st.title("💳 Credit Risk Intelligence Engine")
st.caption("Automated real-time loan decisioning and demographic portfolio risk assessment.")

# Create tabs at the header
tab1, tab2 = st.tabs(["💳 Dynamic Risk Assessment", "📊 Portfolio Analytics (EDA)"])

# TAB 1: Dynamic Risk Assessment
with tab1:
    st.subheader("Applicant Metrics")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sex", options=list(sex_enc.classes_))
        job = st.selectbox("Job Level (0: Unskilled to 3: Highly Skilled)", options=[0, 1, 2, 3], index=2)

    with col2:
        credit_amt = st.number_input("Credit Amount ($)", min_value=100, max_value=20000, value=2500, step=100)
        duration = st.number_input("Duration (Months)", min_value=1, max_value=72, value=12)
        housing = st.selectbox("Housing Status", options=list(housing_enc.classes_))

    with col3:
        saving_acc = st.selectbox("Saving Accounts", options=list(saving_enc.classes_))
        checking_acc = st.selectbox("Checking Account Status", options=list(checking_enc.classes_))

    st.markdown("---")
    
    if st.button("Evaluate Credit Risk"):
        # Categorical Encoding
        sex_val = sex_enc.transform([sex])[0]
        housing_val = housing_enc.transform([housing])[0]
        saving_val = saving_enc.transform([saving_acc])[0]
        checking_val = checking_enc.transform([checking_acc])[0]

        # Feature Vector Formulation
        input_data = pd.DataFrame([[
            age, sex_val, job, housing_val, saving_val, checking_val, credit_amt, duration
        ]], columns=['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration'])

        # Inference
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Display Decision Output
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### Risk Classification")
            if prediction == 0 or str(prediction).lower() == 'good':
                st.markdown("<p class='status-good'>LOW RISK (GOOD CREDIT)</p>", unsafe_allow_html=True)
                confidence = probabilities[0] if len(probabilities) > 1 else 1.0
                st.write(f"Model Confidence: **{confidence * 100:.1f}%**")
            else:
                st.markdown("<p class='status-bad'>HIGH RISK (BAD CREDIT)</p>", unsafe_allow_html=True)
                confidence = probabilities[1] if len(probabilities) > 1 else 1.0
                st.write(f"Model Confidence: **{confidence * 100:.1f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### Automated Decision Advisory")
            
            if prediction == 0 or str(prediction).lower() == 'good':
                st.success("✅ **Decision: Approved**")
                st.markdown("""
                * **Terms:** Standard baseline interest rates applied.
                * **Limits:** Credit volume satisfies risk thresholds.
                * **Verification:** Identity verification required prior to disbursement.
                """)
            else:
                st.error("⚠️ **Decision: Restructure / Decline**")
                st.markdown("""
                * **Mitigation:** Elevated default risk detected.
                * **Alternative:** Require additional collateral or guarantor.
                * **Escalation:** Route to secondary underwriting for manual review.
                """)
            st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: Cleaned Dark Portfolio Analytics
with tab2:
    st.subheader("Historical Portfolio Benchmarks")
    
    # Portfolio Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applicants", len(df))
    m2.metric("Avg Credit Amount", f"${df['Credit amount'].mean():,.0f}")
    m3.metric("Avg Loan Duration", f"{df['Duration'].mean():.1f} Mos")
    m4.metric("Avg Applicant Age", f"{df['Age'].mean():.1f} Yrs")

    st.markdown("---")

    # Dark Theme Seaborn Plots
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("##### Credit Amount Distribution by Housing")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.boxplot(data=df, x='Housing', y='Credit amount', palette=['#3B82F6', '#10B981', '#6366F1'], ax=ax)
        ax.set_ylabel("Credit Amount ($)")
        sns.despine()
        st.pyplot(fig)

    with fig_col2:
        st.markdown("##### Age vs. Credit Amount Overview")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.scatterplot(data=df, x='Age', y='Credit amount', hue='Sex', palette=['#3B82F6', '#EC4899'], alpha=0.8, ax=ax)
        ax.set_ylabel("Credit Amount ($)")
        sns.despine()
        st.pyplot(fig)
