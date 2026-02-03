# Automatic Support Ticket Classification

## Overview
Classifies customer complaints into predefined categories using NLP and ML.

## Features
- Text preprocessing and cleaning
- Topic modeling using NMF
- Supervised classification (LR, LSVC, XGBoost)
- Confidence-aware predictions
- Interactive Streamlit dashboard

## Data Versioning
This project uses DVC to version large datasets.
To download the data locally:
```bash
pip install dvc
dvc pull
```

## Tech Stack
- Python
- scikit-learn
- NLP (TF-IDF, NMF)
- Streamlit

## Project Structure
<tree>

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Results
Model comparison table

## Future Work
- Embeddings-based models
- Multi-domain classification
