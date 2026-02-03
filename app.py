## MORE STYLING 

## =========================
## IMPORTING LIBRARIES
## =========================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import joblib
import streamlit as st
import streamlit.components.v1 as components


if "results_df" not in st.session_state:
    st.session_state.results_df = None

## =========================
## STREAMLIT CONFIG
## =========================
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="📩",
    layout="centered"
)

## =========================
## LOAD MODEL (CACHED)
## =========================
@st.cache_resource
def load_model():
    artifact = joblib.load("Artifacts/lr_artifact.pkl")
    return artifact

artifact = load_model()
model = artifact["model"]
label_map = artifact["label_map"]

## =========================
## HELPER FUNCTIONS
## =========================
def confidence_color(score):
    if score >= 0.8:
        return "#22c55e"   # green
    elif score >= 0.6:
        return "#facc15"   # yellow
    else:
        return "#ef4444"   # red

## =========================
## APP HEADER
## =========================
st.title("📩 Ticket Classifier")
st.write(
    "Enter customer complaints below and the system will automatically "
    "classify them into the appropriate category."
)

st.info("💡 **Tip:** Enter each complaint on a new line to classify them individually.")

## =========================
## USER INPUT
## =========================
user_text = st.text_area(
    "Customer Complaint",
    height=180,
    placeholder=(
        "e.g.\n"
        "My account was debited but the transaction failed\n"
        "Credit card was charged twice\n"
        "I want to open a salary account"
    )
)

## =========================
## PREDICTION
## =========================
if st.button("🔍 Predict Category"):
    if user_text.strip() == "":
        st.warning("Please enter at least one complaint.")
    else:
        complaints = [c.strip() for c in user_text.split("\n") if c.strip()]
        results = []

        for complaint in complaints:
            pred_class = model.predict([complaint])[0]
            proba = model.predict_proba([complaint])[0]

            label = label_map[pred_class]
            confidence = float(np.max(proba))

            results.append({
                "Complaint": complaint,
                "Predicted Category": label,
                "Confidence": confidence
            })

        st.session_state.results_df = pd.DataFrame(results)

results_df = st.session_state.results_df

if results_df is not None:
    st.subheader("📊 Results")


    ## =========================
    ## SUMMARY SECTION
    ## =========================
    st.subheader("📊 Classification Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Complaints", len(results_df))
    col2.metric(
        "High Confidence (≥ 0.8)",
        (results_df["Confidence"] >= 0.8).sum()
    )
    col3.metric(
        "Low Confidence (< 0.6)",
        (results_df["Confidence"] < 0.6).sum()
    )

    st.markdown("---")

    show_table = st.checkbox("📋 Show full results as table")

    if show_table:
        styled_df = (
            results_df
            .assign(Confidence=lambda x: x["Confidence"].round(2))
            .style
            .background_gradient(cmap="Blues", subset=["Confidence"])
            .set_table_styles([
                {
                    'selector': 'th',
                    'props': [
                        ('background-color', '#1e293b'),
                        ('color', 'white'),
                        ('font-weight', 'bold')
                    ]
                }
            ])
    )

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=350
        )


    ## =========================
    ## CARD-BASED RESULTS
    ## =========================
    MAX_CARDS = 3
    st.subheader("🧾 Highlighted Results")

    for _, row in results_df.head(MAX_CARDS).iterrows():
        color = confidence_color(row["Confidence"])

        card_html = f"""
                <div style="
                    /* Remove the solid hex line so the RGBA transparency works */
                    background-color: rgba(30, 41, 59, 0.7); 
                    backdrop-filter: blur(10px);             
                    -webkit-backdrop-filter: blur(10px);    /* Added for Safari support */
                    border: 1px solid rgba(255, 255, 255, 0.1); 
                    padding: 20px;
                    border-radius: 16px;
                    margin-bottom: 16px;
                    border-left: 6px solid {color};
                    box-shadow: 0 6px 18px rgba(0,0,0,0.55);
                ">
                    <p style="color:#94a3b8;font-size:13px;margin-bottom:6px;">
                        <b>Complaint</b>
                    </p>
                    <p style="color:#ffffff;font-size:15px;margin-bottom:10px;">
                        {row['Complaint']}
                    </p>

                    <p style="margin-top:12px;">
                        <b style="color:#94a3b8;">🏷️ Category:</b>
                        <span style="color:#38bdf8;font-weight:600;">
                            {row['Predicted Category']}
                        </span>
                    </p>

                    <p style="margin-bottom:6px;">
                        <b style="color:#94a3b8;">📊 Confidence:</b> 
                        <span style="color:#38bdf8;font-weight:600;">
                            {row['Confidence']:.2f}
                        </span>
                    </p>
                </div>
        """
        components.html(card_html,height=220)

        st.progress(row["Confidence"])

        if row["Confidence"] < 0.6:
            st.warning(
                "⚠️ Low confidence prediction — manual review recommended."
            )


