## IMPORTING LIBRARIES
import json
import re, string
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk, spacy
import nbformat

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import NMF

## IMPORT REQUIRED LIBRARIES
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import joblib
import streamlit as st

## STREAMLIT APP
st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    artifact = joblib.load("Artifacts/lr_artifact.pkl")
    return artifact

artifact = load_model()
model = artifact["model"]
label_map = artifact["label_map"]


st.title("Ticket Classifier")
st.write(
    "Enter Customer complaints below and the system will automatically "
    "classify it into the appropriate category."
)
st.info("💡 **Tip:** Enter each complaint on a new line to classify them individually.")

## TAKING COMPLAINT INPUT FROM THE USER
user_text = st.text_area(
    "Customer Complaint",
    height=150,
    placeholder="e.g. My account was debited but the transaction failed..."
)

## PREDICT BUTTON
if st.button("Predict Category"):
    if user_text.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        complaints = [c.strip() for c in user_text.split("\n") if c.strip()]
        
        results = []

        # 2. Loop through each complaint
        for complaint in complaints:
            # Predict class and probabilities
            pred_class = model.predict([complaint])[0]
            proba = model.predict_proba([complaint])[0]
            
            # Get label and confidence
            label = label_map[pred_class]
            confidence = np.max(proba)
            
            results.append({
                "Complaint Snippet": complaint[:50] + "...", 
                "Predicted Category": label,
                "Confidence": round(confidence, 2)
            })

        results_df = pd.DataFrame(results)
        st.subheader("📊 Classification Results")
        # Apply styling
        styled_df = results_df.style.background_gradient(cmap='Blues', subset=['Confidence']) \
                                    .set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}, subset=['Complaint Snippet', 'Predicted Category'])

        st.dataframe(styled_df, use_container_width=True)




