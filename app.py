# 1 Good (Lower Risk) 0 Bad (Higher Risk)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration for clean layout
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="centered")

# Create tabs at the header
tab1, tab2 = st.tabs(["💳 Credit Risk Predictor", "📊 Data Analytics Dashboard"])

with tab1:
    ## Best fit model import
    model = joblib.load("extra_trees_credit_model.pkl")
    encoders = {col : joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

    ## Header section with title and subtext
    st.title("💳 Credit Risk Predictor")
    st.caption("Enter applicant details below to evaluate creditworthiness and default risk.")
    st.divider()

    ## Prepare the columns for app (organise into two clean UI columns)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Personal Details")
        age = st.number_input(
            "Age", min_value = 18, max_value = 80, value = 30,
            help="Applicant's current age in years (must be between 18 and 80)."
        )
        sex = st.selectbox(
            "Sex", ["male", "female"],
            help="Applicant's legal gender designation."
        )
        job = st.number_input(
            "Job (0-3)", min_value = 0, max_value = 3, value = 1,
            help="0: Unskilled / non-resident, 1: Unskilled / resident, 2: Skilled employee / official, 3: Highly skilled / management."
        )
        housing = st.selectbox(
            "Housing", ["own", "rent", "free"],
            help="Current residential living status: 'own' (homeowner), 'rent' (tenant), or 'free' (provided accommodation)."
        )

    with col2:
        st.subheader("💰 Financial Details")
        saving_accounts = st.selectbox(
            "Saving accounts", ["rich", "quite rich", "moderate", "little"],
            help="""Estimated total funds in savings accounts:
        - little: < 100 DM
        - moderate: 100 to 500 DM
        - quite rich: 500 to 1000 DM
        - rich: ≥ 1000 DM"""
            )
        checking_account = st.selectbox(
            "Checking account", ["moderate", "little", "rich"],
            help="""Current liquid balance in checking account:
        - little: < 0 DM / minimal balance
        - moderate: 0 to 200 DM
        - rich: ≥ 200 DM"""
            )
        credit_amount = st.number_input(
            "Credit amount", min_value = 0, value = 1000,
            help="Total principal loan amount requested by the applicant."
        )
        duration = st.number_input(
            "Duration (months)", min_value = 1, value = 12,
            help="Total duration in months requested to repay the full credit amount."
        )

    st. divider()

    ## Prepare input for the mdoel
    input_df = pd.DataFrame({
        "Age" : [age],
        "Sex" : [encoders["Sex"].transform([sex])[0]],
        "Job" : [job],
        "Housing" : [encoders["Housing"].transform([housing])[0]],
        "Saving accounts" : [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account" : [encoders["Checking account"].transform([checking_account])[0]],
        "Credit amount" : [credit_amount],
        "Duration" : [duration]
    })

    ## Prediction button
    if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
        pred = model.predict(input_df)[0]

        if pred ==1:
            st.success("### 🎉 The predicted credit risk is: **GOOD**")
            st.toast("Applicant meets credit criteria!", icon="✅")
        else:
            st.error("### ⚠️ The predicted credit risk is: **BAD**")

    # styled suggestions containers
            with st.container(border=True):
                st.markdown("#### 💡 Suggestions to Improve Credit Approval")

                #Dynamics rules based on inputs
                if duration <= 6 and credit_amount > 1000:
                    st.write("• **Increase Loan Duration:** Paying back over a longer period lowers monthly installment pressure.")

                if credit_amount > 3000:
                    st.write("• **Reduce Credit Amount:** Requesting a smaller amount decreases overall default exposure.")

                if saving_accounts == "little":
                    st.write("• **Increase Savings Balance:** A higher savings account balance provides financial backing.")

                if checking_account == "little":
                    st.write("• **Maintain Higher Checking Reserve:** Maintaining more liquidity in the checking account improves credit rating.")

                if housing in ["rent", "free"]:
                    st.write("• **Provide Guarantor/Collateral:** Adding a co-signer or proof of property can offset housing risk.")

with tab2:
    st.header("Dataset Analytics & Risk Factors")

   # @st.cache_data
    def load_data():
        return pd.read_csv("german_credit_data.csv")

    df = load_data()
    sns.set_theme(style="whitegrid")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Credit Risk Breakdown")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.countplot(
            data = df,
            x = "Risk",
            hue = "Risk",
            palette = {"good": "#22c55e", "bad": "#ef4444"},
            legend = False,
            ax = ax1,
        )
        st.pyplot(fig1)

    with col2:
        st.subheader("Credit Amount vs Duration")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(
            data=df,
            x="Duration",
            y="Credit amount",
            hue="Risk",
            palette={"good": "#22c55e", "bad": "#ef4444"},
            alpha=0.7,
            ax=ax2,
        )
        st.pyplot(fig2)

# End of col1 and col2 blocks above
# 3rd Graph (Outside col2, sitting directly under tab2)
    st.subheader("Loan Purpose Breakdown")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.countplot(
        data=df,
        y="Purpose",
        hue="Risk",
        palette={"good": "#22c55e", "bad": "#ef4444"},
        order=df["Purpose"].value_counts().index,
        ax=ax3,
    )   
    st.pyplot(fig3)