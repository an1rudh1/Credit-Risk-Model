import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Explainable Credit Risk", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    calibrated = joblib.load("credit_risk_model.pkl")   # calibrated XGBoost -> probability
    base = joblib.load("xgb_pipeline.pkl")              # tuned XGBoost pipeline -> SHAP
    threshold = float(joblib.load("best_threshold.pkl"))
    explainer = shap.TreeExplainer(base.named_steps["classifier"])
    return calibrated, base, threshold, explainer


calibrated, base, threshold, explainer = load_artifacts()

st.title("💳 Explainable Credit Risk Assessment")
st.caption(
    "XGBoost model with calibrated probabilities and SHAP explanations. "
    "Built on a public Kaggle credit-risk dataset. Demo only, not financial advice."
)

with st.form("applicant"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 28)
        income = st.number_input("Annual income", 4000, 6_000_000, 60000, step=1000)
        home = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp_len = st.number_input("Employment length (years)", 0, 60, 4)
        hist_len = st.number_input("Credit history length (years)", 2, 40, 5)
    with c2:
        intent = st.selectbox(
            "Loan purpose",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        grade = st.selectbox("Loan grade", list("ABCDEFG"))
        loan_amnt = st.number_input("Loan amount", 500, 35000, 10000, step=500)
        int_rate = st.number_input("Interest rate (%)", 5.0, 24.0, 11.0, step=0.1)
        prior_default = st.selectbox("Previous default on file?", ["N", "Y"])
    submitted = st.form_submit_button("Assess risk", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "person_age": age,
        "person_income": income,
        "person_home_ownership": home,
        "person_emp_length": emp_len,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": round(loan_amnt / income, 2),
        "cb_person_default_on_file": prior_default,
        "cb_person_cred_hist_length": hist_len,
    }])

    prob = float(calibrated.predict_proba(row)[:, 1][0])
    high_risk = prob >= threshold

    st.subheader("Result")
    m1, m2 = st.columns(2)
    m1.metric("Default probability", f"{prob:.1%}")
    m2.metric("Decision threshold", f"{threshold:.1%}")
    if high_risk:
        st.error("High risk: recommend manual review or rejection.")
    else:
        st.success("Low risk: recommend approval.")
    st.progress(min(max(prob, 0.0), 1.0))

    # ---- SHAP explanation for this applicant ----
    X_t = base.named_steps["preprocessor"].transform(row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    names = [n.split("__", 1)[-1] for n in base.named_steps["preprocessor"].get_feature_names_out()]
    sv = explainer.shap_values(X_t)[0]

    contrib = pd.Series(sv, index=names)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]

    st.subheader("Why this score?")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index, top.values, color=["#d9534f" if v > 0 else "#2e8b57" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Impact on risk (SHAP, log-odds)")
    st.pyplot(fig)
    st.caption("Red bars push the applicant toward default, green bars push toward repayment.")
