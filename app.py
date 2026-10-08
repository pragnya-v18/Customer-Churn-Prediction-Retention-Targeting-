"""Customer Churn Risk Predictor — Streamlit app.

Enter a customer's details -> get a calibrated churn probability, an expected-value
decision on whether a retention offer is worth it (business assumptions are editable),
and the SHAP reasons behind the score with suggested retention actions.
Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")

MODEL_PATH = Path(__file__).parent / "churn_model.joblib"
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
CAT_PREFIXES = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"]

# Feature -> suggested retention action (shown only when that feature pushes risk UP)
ACTIONS = {
    "Contract_Month-to-month": "Offer an incentive to move to a 1- or 2-year contract.",
    "tenure": "Early-life customer: schedule onboarding check-ins in the first 6 months.",
    "OnlineSecurity_No": "Offer a free trial of Online Security.",
    "TechSupport_No": "Offer a free trial of Tech Support.",
    "PaymentMethod_Electronic check": "Nudge towards automatic payment (small bill credit).",
    "InternetService_Fiber optic": "Check fiber service quality / pricing complaints for this customer.",
    "MonthlyCharges": "Review the plan: offer a right-sized or discounted plan.",
    "PaperlessBilling_Yes": "Send a proactive 'value for money' summary with the bill.",
    "num_addons": "Bundle add-on services (security, backup, support) at a discount.",
}


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Must match the feature engineering used in the notebook."""
    d = d.copy()
    d["num_addons"] = (d[ADDON_COLS] == "Yes").sum(axis=1)
    d["is_autopay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    d["has_family"] = ((d["Partner"] == "Yes") | (d["Dependents"] == "Yes")).astype(int)
    return d


@st.cache_resource
def load_artifacts():
    art = joblib.load(MODEL_PATH)
    pipe = art["pipeline"]                      # tuned XGBoost pipeline (for SHAP)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["clf"]
    return (art["calibrated"], prep, shap.TreeExplainer(model),
            art["ev_params"], art["feature_columns"])


def pretty(feature: str, raw_value: float, row: pd.Series) -> str:
    """Human-readable label for a model feature."""
    for col in CAT_PREFIXES:
        if feature.startswith(col + "_"):
            val = feature[len(col) + 1:]
            return f"{col} = {val}" if raw_value >= 0.5 else f"{col} ≠ {val}"
    if feature in row.index:
        v = row[feature]
        return f"{feature} = {v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"{feature} = {v}"
    return feature


calibrated, prep, explainer, EV, FEATURE_COLUMNS = load_artifacts()

# ----------------------------- Sidebar inputs -----------------------------
with st.sidebar:
    st.header("Customer profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 5)

    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "No":
        multiple = "No phone service"
        st.caption("Multiple lines: No phone service")
    else:
        multiple = st.selectbox("Multiple lines", ["No", "Yes"])

    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addons = {}
    for col in ADDON_COLS:
        if internet == "No":
            addons[col] = "No internet service"
        else:
            addons[col] = st.selectbox(col, ["No", "Yes"], key=col)
    if internet == "No":
        st.caption("Add-ons: No internet service")

    st.subheader("Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=85.0, step=0.5)
    total = round(tenure * monthly, 2)
    st.caption(f"Total charges (estimated = tenure × monthly): ${total:,.2f}")

    st.subheader("Business assumptions")
    offer_cost = st.number_input("Offer cost per customer ($)", 0.0, 500.0, float(EV["offer_cost"]), 5.0)
    save_rate = st.slider("Save rate (offer retains a would-be churner)", 0.0, 1.0, float(EV["save_rate"]), 0.05)
    clv_months = st.slider("Months of revenue preserved if retained", 1, 36, int(EV["clv_months"]))

raw = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multiple, "InternetService": internet, **addons,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}])
row = add_features(raw)[FEATURE_COLUMNS]

# ----------------------------- Prediction -----------------------------
prob = float(calibrated.predict_proba(row)[0, 1])
revenue_at_risk = monthly * clv_months
expected_gain = prob * save_rate * revenue_at_risk - offer_cost
flag = expected_gain > 0
band = "High" if prob >= 0.5 else ("Elevated" if prob >= 0.2 else "Low")

st.title("📉 Customer Churn Risk Predictor")
st.caption("Tuned XGBoost model trained on the IBM Telco dataset · calibrated probabilities · explanations via SHAP")

c1, c2, c3 = st.columns(3)
c1.metric("Churn probability", f"{prob:.0%}")
c2.metric("Risk band", band)
c3.metric("Retention offer worth it?", "✅ Yes — contact" if flag else "❌ No",
          delta=f"expected net ${expected_gain:,.0f}", delta_color="normal" if flag else "inverse")
st.progress(min(max(prob, 0.0), 1.0))
st.caption(
    "Decision rule: contact if  P(churn) × save rate × revenue at risk  >  offer cost  "
    f"→  {prob:.0%} × {save_rate:.0%} × ${revenue_at_risk:,.0f} − ${offer_cost:,.0f} = ${expected_gain:,.0f}. "
    "Probabilities are isotonic-calibrated on training data; change the business assumptions in the sidebar."
)

# ----------------------------- SHAP explanation -----------------------------
X_t = prep.transform(row)
shap_vals = explainer.shap_values(X_t)[0]
names = prep.get_feature_names_out()
x_vals = np.asarray(X_t)[0]

contrib = pd.DataFrame({
    "feature": names,
    "label": [pretty(n, v, row.iloc[0]) for n, v in zip(names, x_vals)],
    "shap": shap_vals,
}).sort_values("shap", key=np.abs, ascending=False)

st.subheader("Why this score?")
st.caption("SHAP shows what pushes the raw model score up or down for this customer; the probability above is the calibrated version of that score.")
left, right = st.columns([3, 2])

with left:
    top = contrib.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(top["label"], top["shap"], color=["#C44E52" if v > 0 else "#4C72B0" for v in top["shap"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Impact on churn risk (SHAP, log-odds)   ← lowers | raises →")
    ax.set_title("Top 10 factors for this customer")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    ups = contrib[contrib["shap"] > 0].head(3)
    downs = contrib[contrib["shap"] < 0].head(3)
    st.markdown("**🔺 Raising churn risk**")
    for _, r in ups.iterrows():
        st.markdown(f"- {r['label']}  (+{r['shap']:.2f})")
    st.markdown("**🔻 Lowering churn risk**")
    for _, r in downs.iterrows():
        st.markdown(f"- {r['label']}  ({r['shap']:.2f})")

st.subheader("Suggested retention actions")
actions = []
for _, r in contrib[contrib["shap"] > 0].iterrows():
    if r["feature"] in ACTIONS and ACTIONS[r["feature"]] not in actions:
        actions.append(ACTIONS[r["feature"]])
    if len(actions) == 3:
        break
if actions and flag:
    for a in actions:
        st.markdown(f"- {a}")
elif flag:
    st.info("No rule-based action matches this customer's top drivers; review the factors above.")
else:
    st.success("Expected benefit of an offer is below its cost — no retention offer needed right now.")

with st.expander("Model input (after feature engineering)"):
    st.dataframe(row.T.astype(str).rename(columns={0: "value"}), width="stretch")
