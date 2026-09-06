# Customer Segmentation MLOps Project

An end-to-end customer segmentation system built using the Online Retail II dataset.

The project transforms transaction-level retail data into customer-level behavioral features and uses K-Means clustering to identify meaningful customer segments.

## Live Demo

> **Live Demo:** [Add Streamlit URL here]

## Model

> **Hugging Face Model:** https://huggingface.co/ZubairAli25266/customer_segementation

The production model and metadata are hosted publicly on Hugging Face.

The Streamlit application downloads these artifacts at runtime instead of storing the model binary inside the GitHub repository.

## Business Segments

The final model produces two customer segments:

### Active / High-Value Customers

Customers with:

- Low recency
- Higher purchase frequency
- Higher monetary value
- Higher total quantity
- Greater product diversity
- Higher average order value

This segment represents the majority of business revenue.

### Inactive / Low-Value Customers

Customers with:

- High recency
- Low purchase frequency
- Lower monetary value
- Lower purchase quantity
- Lower product diversity
- Lower average order value

This segment represents customers who may require re-engagement strategies.

## Dataset

The project uses the Online Retail II dataset.

The production pipeline:

1. Loads both Excel sheets.
2. Removes duplicate transactions.
3. Removes cancellation invoices.
4. Removes invalid quantities and prices.
5. Removes transactions without customer IDs.
6. Restricts analysis to United Kingdom customers.
7. Aggregates transactions at customer level.

The raw dataset is intentionally not committed to GitHub because it is a large data artifact.

## Customer Features

Six behavioral features are used:

- Recency
- Frequency
- Monetary
- Total Quantity
- Unique Products
- Average Order Value

Five heavily right-skewed features are transformed using `log1p`:

- Frequency
- Monetary
- Total Quantity
- Unique Products
- Average Order Value

Recency remains untransformed.

The resulting features are standardized using `StandardScaler`.

## Model Selection

Several clustering algorithms were evaluated:

- K-Means
- Agglomerative Clustering
- Gaussian Mixture Model
- DBSCAN

K-Means with two clusters was selected based on clustering metrics and business interpretability.

### Final Full-Dataset Metrics

| Metric | Score |
|---|---:|
| Silhouette Score | 0.3632 |
| Davies-Bouldin Index | 1.0260 |
| Calinski-Harabasz Score | 4151.84 |

### Holdout Evaluation

The production training pipeline uses a customer-level 80/20 split.

| Metric | Holdout Score |
|---|---:|
| Silhouette Score | 0.3425 |
| Davies-Bouldin Index | 1.0832 |
| Calinski-Harabasz Score | 734.60 |

## Architecture

```text
Online Retail II
       |
       v
Data Loader
       |
       v
Data Cleaning
       |
       v
Customer Aggregation
       |
       v
Feature Engineering
       |
       +--> log1p transformations
       |
       +--> StandardScaler
       |
       v
K-Means
       |
       v
Cluster Profiling
       |
       v
Business Interpretation
       |
       +-------------------+
       |                   |
       v                   v
Model Artifact        Metadata
       |                   |
       +---------+---------+
                 |
                 v
        Hugging Face Hub
                 |
                 v
         Streamlit App
                 |
                 v
       Customer Prediction



customer-segmentation/
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── configs/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_understaing_data_and_eda.ipynb
│   ├── 02_customer_feature_engineering.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_final_model_selection.ipynb
│
├── pipelines/
│   ├── prediction_pipeline.py
│   └── training_pipeline.py
│
├── src/
│   └── customer_segmentation/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── utils/
│
├── tests/
│   ├── test_prediction_pipeline.py
│   └── test_profiling.py
│
├── .gitignore
├── README.md
└── requirements.txt