# Customer Segmentation MLOps

An end-to-end customer segmentation project on the **Online Retail II** dataset — from raw transactions to a deployed prediction app.

The pipeline cleans transaction-level data, engineers customer-level behavioral features, evaluates multiple clustering algorithms, validates the selected model on a holdout set, and serves predictions through a Streamlit app backed by a model hosted on Hugging Face.

**Live Demo:** https://customersegementation-t5s8rntqez9a2bdbahe2b8.streamlit.app/
**Model:** https://huggingface.co/ZubairAli25266/customer_segementation

---

## Project Objective

Identify meaningful customer segments based on purchasing behavior, using unsupervised clustering on customer-level features derived from transaction history. The final model separates customers into two segments:

- Active / High-Value Customers
- Inactive / Low-Value Customers

---

## Dataset

**Online Retail II** — two Excel sheets (2009–2010, 2010–2011), combined into a single raw dataset:

- 1,067,371 transactions, 8 columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`

Raw data is excluded from the repo due to size.

**Cleaning steps:** combine sheets → remove duplicates → remove cancellations → remove non-positive quantity/price rows → remove missing Customer IDs → restrict to UK customers.

Result: **700,388 valid UK transactions, 5,350 unique customers** (`src/customer_segmentation/data/cleaner.py`).

---

## Feature Engineering

Transaction data is aggregated into a **5,350 × 6** customer-level feature matrix:

| Feature | Definition |
|---|---|
| Recency | Days since most recent purchase |
| Frequency | Number of unique orders |
| Monetary | Total revenue (Quantity × Price, summed) |
| Total Quantity | Total units purchased |
| Unique Products | Distinct products purchased |
| Average Order Value | Monetary / Frequency |

**Transformation:** Frequency, Monetary, Total Quantity, Unique Products, and Average Order Value were right-skewed and log-transformed (`log1p`); Recency was left as-is (skew 0.87). All features were then standardized with `StandardScaler`.

| Feature | Skew (before) | Skew (after) |
|---|---:|---:|
| Average Order Value | 58.18 | -0.07 |
| Monetary | 28.19 | 0.25 |
| Total Quantity | 16.31 | -0.09 |
| Frequency | 10.43 | 0.97 |
| Unique Products | 5.17 | -0.28 |
| Recency | 0.87 | 0.87 |

---

## Clustering Algorithms Evaluated

K-Means, Agglomerative Clustering, Gaussian Mixture Model, and DBSCAN, compared on Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, cluster balance, revenue/quantity/order contribution, and business interpretability.

K-Means was tested across K=2 through K=10; K=2 was the strongest configuration.

**Algorithm comparison (best configuration each):**

| Algorithm | Config | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---:|---:|
| **K-Means** | K=2 | **0.3632** | **1.0260** | **4151.84** |
| Agglomerative | K=2 | 0.3455 | 1.0708 | 3632.32 |
| Gaussian Mixture | K=2 | 0.3061 | 1.1608 | 3104.39 |
| DBSCAN | eps=0.5 | 0.1713 | 2.9180 | 804.75 |

K-Means (K=2) was selected — highest Silhouette, lowest Davies-Bouldin, highest Calinski-Harabasz, and well-balanced clusters (50.73% / 49.27% customer split).

---

## Cluster Profiles

| Feature | Cluster 0 (Inactive/Low-Value) | Cluster 1 (Active/High-Value) |
|---|---:|---:|
| Recency | 326 days | 37 days |
| Frequency | 1 order | 7 orders |
| Monetary | 334.22 | 2,182.70 |
| Total Quantity | 183.5 | 1,305.5 |
| Unique Products | 20 | 101.5 |
| Average Order Value | 197.68 | 337.72 |

**Relative to overall median:**

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Recency | 3.31× | 0.38× |
| Frequency | 0.33× | 2.33× |
| Monetary | 0.40× | 2.63× |
| Total Quantity | 0.40× | 2.81× |
| Unique Products | 0.45× | 2.31× |
| Average Order Value | 0.73× | 1.25× |

---

## Business Contribution

Despite near-equal customer counts, the two clusters differ sharply in business value:

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customer Share | 50.73% | 49.27% |
| Revenue Share | 7.65% | 92.35% |
| Quantity Share | 7.27% | 92.73% |
| Order Share | 15.04% | 84.96% |
| Avg. Revenue/Customer | 405.61 | 5,041.12 |
| Avg. Orders/Customer | 1.86 | 10.81 |

Roughly half of customers (Cluster 1) generate over 92% of total revenue — about 12.4× more revenue and 5.8× more orders per customer than Cluster 0.

---

## PCA (Visualization Only)

PCA was used only to visualize separation, not as clustering input. PC1 (67.30% variance) and PC2 (15.46%) together explain 82.76% of variance; separation occurs primarily along PC1.

---

## Final Model Selection

**K-Means, K=2**, selected on the combination of clustering metrics, cluster balance, behavioral profiles, revenue contribution, and interpretability.

Cluster ID → business label mapping is stored in model metadata rather than hard-coded, since K-Means cluster IDs are arbitrary:

```text
0 → Active / High-Value Customers
1 → Inactive / Low-Value Customers
```

---

## Holdout Validation

An 80/20 customer-level train/holdout split was used. The model was fit only on training customers; holdout customers were transformed and assigned to clusters using the fitted pipeline.

- Training: 4,280 customers | Holdout: 1,070 customers | Total: 5,350

| Metric | Holdout Score |
|---|---:|
| Silhouette Score | 0.3425 |
| Davies-Bouldin Index | 1.0832 |
| Calinski-Harabasz Score | 734.60 |

Holdout results are consistent with the full-dataset experiment.

---

## Production Pipeline

Preprocessing and clustering are packaged into a single scikit-learn pipeline (log transform → StandardScaler → K-Means) so training-time and inference-time preprocessing can't diverge. Model metadata stores the model config, features, holdout metrics, and cluster-to-label mapping; the prediction pipeline reads this mapping rather than assuming cluster meaning.

**Prediction flow:** customer input → feature validation → pipeline (log1p → scale → K-Means) → cluster ID → metadata lookup → business segment.

---

## Architecture

```text
Notebooks (EDA, feature engineering, clustering, model selection)
        |
Production Code (loading, cleaning, aggregation, feature processing, pipeline, evaluation)
        |
Model Artifacts (pipeline + metadata)
        |
Hugging Face Hub (production model + metadata)
        |
Streamlit App (input -> prediction -> segment)
```

---

## Project Structure

```text
customer-segmentation/
├── app/streamlit_app.py
├── artifacts/{metrics,plots,reports}/
├── configs/
├── data/{raw,processed}/
├── models/
├── notebooks/
│   ├── 01_understanding_data_and_eda.ipynb
│   ├── 02_customer_feature_engineering.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_final_model_selection.ipynb
├── pipelines/
│   ├── prediction_pipeline.py
│   └── training_pipeline.py
├── src/customer_segmentation/
│   ├── data/{loader.py, cleaner.py, aggregator.py}
│   ├── evaluation/profiling.py
│   ├── features/transformer.py
│   └── models/{clustering.py, pipeline.py, metadata.py}
├── tests/
│   ├── test_prediction_pipeline.py
│   └── test_profiling.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Testing

Automated tests cover feature validation, missing-feature/value detection, metadata loading, cluster profiling, and cluster mapping.

```bash
pytest -q
```

---

## Streamlit Application

Users input the six customer features; the app loads the model and metadata from Hugging Face, runs the preprocessing pipeline, predicts the cluster, maps it to a business segment, and displays the result.

---

## Deployment

Deployed on Streamlit Community Cloud, pulling the model and metadata from Hugging Face Hub at runtime. Dependencies are listed in `requirements.txt`.

```bash
git clone https://github.com/zaza25266/customer_segementation.git
cd customer_segementation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
streamlit run app/streamlit_app.py
```

---

## Limitations

This is unsupervised learning — there are no ground-truth labels, so metrics measure cluster structure rather than prediction accuracy. A Silhouette Score of 0.3632 indicates moderate, not perfect, separation. Cluster labels are business interpretations, not known classes, and customer behavior can shift over time; a continuously operating system would need periodic retraining and monitoring.

---

## Future MLOps Improvements

MLflow tracking, DVC data versioning, Great Expectations / data validation, Evidently drift monitoring, automated retraining, GitHub Actions CI/CD, Docker, AWS and Kubernetes deployment, scheduled training, and model/data quality monitoring.

---

## Technologies

**ML:** Python, Pandas, NumPy, Scikit-learn, K-Means, Agglomerative Clustering, GMM, DBSCAN, PCA
**Production:** Joblib, Pytest, Hugging Face Hub, Streamlit
**Dev:** Git, GitHub, virtual environments

---

## Links

- **Live Demo:** # Customer Segmentation MLOps

An end-to-end customer segmentation project on the **Online Retail II** dataset — from raw transactions to a deployed prediction app.

The pipeline cleans transaction-level data, engineers customer-level behavioral features, evaluates multiple clustering algorithms, validates the selected model on a holdout set, and serves predictions through a Streamlit app backed by a model hosted on Hugging Face.

**Live Demo:** `PASTE YOUR STREAMLIT LIVE URL HERE`
**Model:** https://huggingface.co/ZubairAli25266/customer_segementation

---

## Project Objective

Identify meaningful customer segments based on purchasing behavior, using unsupervised clustering on customer-level features derived from transaction history. The final model separates customers into two segments:

- Active / High-Value Customers
- Inactive / Low-Value Customers

---

## Dataset

**Online Retail II** — two Excel sheets (2009–2010, 2010–2011), combined into a single raw dataset:

- 1,067,371 transactions, 8 columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`

Raw data is excluded from the repo due to size.

**Cleaning steps:** combine sheets → remove duplicates → remove cancellations → remove non-positive quantity/price rows → remove missing Customer IDs → restrict to UK customers.

Result: **700,388 valid UK transactions, 5,350 unique customers** (`src/customer_segmentation/data/cleaner.py`).

---

## Feature Engineering

Transaction data is aggregated into a **5,350 × 6** customer-level feature matrix:

| Feature | Definition |
|---|---|
| Recency | Days since most recent purchase |
| Frequency | Number of unique orders |
| Monetary | Total revenue (Quantity × Price, summed) |
| Total Quantity | Total units purchased |
| Unique Products | Distinct products purchased |
| Average Order Value | Monetary / Frequency |

**Transformation:** Frequency, Monetary, Total Quantity, Unique Products, and Average Order Value were right-skewed and log-transformed (`log1p`); Recency was left as-is (skew 0.87). All features were then standardized with `StandardScaler`.

| Feature | Skew (before) | Skew (after) |
|---|---:|---:|
| Average Order Value | 58.18 | -0.07 |
| Monetary | 28.19 | 0.25 |
| Total Quantity | 16.31 | -0.09 |
| Frequency | 10.43 | 0.97 |
| Unique Products | 5.17 | -0.28 |
| Recency | 0.87 | 0.87 |

---

## Clustering Algorithms Evaluated

K-Means, Agglomerative Clustering, Gaussian Mixture Model, and DBSCAN, compared on Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, cluster balance, revenue/quantity/order contribution, and business interpretability.

K-Means was tested across K=2 through K=10; K=2 was the strongest configuration.

**Algorithm comparison (best configuration each):**

| Algorithm | Config | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---:|---:|---:|
| **K-Means** | K=2 | **0.3632** | **1.0260** | **4151.84** |
| Agglomerative | K=2 | 0.3455 | 1.0708 | 3632.32 |
| Gaussian Mixture | K=2 | 0.3061 | 1.1608 | 3104.39 |
| DBSCAN | eps=0.5 | 0.1713 | 2.9180 | 804.75 |

K-Means (K=2) was selected — highest Silhouette, lowest Davies-Bouldin, highest Calinski-Harabasz, and well-balanced clusters (50.73% / 49.27% customer split).

---

## Cluster Profiles

| Feature | Cluster 0 (Inactive/Low-Value) | Cluster 1 (Active/High-Value) |
|---|---:|---:|
| Recency | 326 days | 37 days |
| Frequency | 1 order | 7 orders |
| Monetary | 334.22 | 2,182.70 |
| Total Quantity | 183.5 | 1,305.5 |
| Unique Products | 20 | 101.5 |
| Average Order Value | 197.68 | 337.72 |

**Relative to overall median:**

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Recency | 3.31× | 0.38× |
| Frequency | 0.33× | 2.33× |
| Monetary | 0.40× | 2.63× |
| Total Quantity | 0.40× | 2.81× |
| Unique Products | 0.45× | 2.31× |
| Average Order Value | 0.73× | 1.25× |

---

## Business Contribution

Despite near-equal customer counts, the two clusters differ sharply in business value:

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customer Share | 50.73% | 49.27% |
| Revenue Share | 7.65% | 92.35% |
| Quantity Share | 7.27% | 92.73% |
| Order Share | 15.04% | 84.96% |
| Avg. Revenue/Customer | 405.61 | 5,041.12 |
| Avg. Orders/Customer | 1.86 | 10.81 |

Roughly half of customers (Cluster 1) generate over 92% of total revenue — about 12.4× more revenue and 5.8× more orders per customer than Cluster 0.

---

## PCA (Visualization Only)

PCA was used only to visualize separation, not as clustering input. PC1 (67.30% variance) and PC2 (15.46%) together explain 82.76% of variance; separation occurs primarily along PC1.

---

## Final Model Selection

**K-Means, K=2**, selected on the combination of clustering metrics, cluster balance, behavioral profiles, revenue contribution, and interpretability.

Cluster ID → business label mapping is stored in model metadata rather than hard-coded, since K-Means cluster IDs are arbitrary:

```text
0 → Active / High-Value Customers
1 → Inactive / Low-Value Customers
```

---

## Holdout Validation

An 80/20 customer-level train/holdout split was used. The model was fit only on training customers; holdout customers were transformed and assigned to clusters using the fitted pipeline.

- Training: 4,280 customers | Holdout: 1,070 customers | Total: 5,350

| Metric | Holdout Score |
|---|---:|
| Silhouette Score | 0.3425 |
| Davies-Bouldin Index | 1.0832 |
| Calinski-Harabasz Score | 734.60 |

Holdout results are consistent with the full-dataset experiment.

---

## Production Pipeline

Preprocessing and clustering are packaged into a single scikit-learn pipeline (log transform → StandardScaler → K-Means) so training-time and inference-time preprocessing can't diverge. Model metadata stores the model config, features, holdout metrics, and cluster-to-label mapping; the prediction pipeline reads this mapping rather than assuming cluster meaning.

**Prediction flow:** customer input → feature validation → pipeline (log1p → scale → K-Means) → cluster ID → metadata lookup → business segment.

---

## Architecture

```text
Notebooks (EDA, feature engineering, clustering, model selection)
        |
Production Code (loading, cleaning, aggregation, feature processing, pipeline, evaluation)
        |
Model Artifacts (pipeline + metadata)
        |
Hugging Face Hub (production model + metadata)
        |
Streamlit App (input -> prediction -> segment)
```

---

## Project Structure

```text
customer-segmentation/
├── app/streamlit_app.py
├── artifacts/{metrics,plots,reports}/
├── configs/
├── data/{raw,processed}/
├── models/
├── notebooks/
│   ├── 01_understanding_data_and_eda.ipynb
│   ├── 02_customer_feature_engineering.ipynb
│   ├── 03_clustering_experiments.ipynb
│   └── 04_final_model_selection.ipynb
├── pipelines/
│   ├── prediction_pipeline.py
│   └── training_pipeline.py
├── src/customer_segmentation/
│   ├── data/{loader.py, cleaner.py, aggregator.py}
│   ├── evaluation/profiling.py
│   ├── features/transformer.py
│   └── models/{clustering.py, pipeline.py, metadata.py}
├── tests/
│   ├── test_prediction_pipeline.py
│   └── test_profiling.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Testing

Automated tests cover feature validation, missing-feature/value detection, metadata loading, cluster profiling, and cluster mapping.

```bash
pytest -q
```

---

## Streamlit Application

Users input the six customer features; the app loads the model and metadata from Hugging Face, runs the preprocessing pipeline, predicts the cluster, maps it to a business segment, and displays the result.

---

## Deployment

Deployed on Streamlit Community Cloud, pulling the model and metadata from Hugging Face Hub at runtime. Dependencies are listed in `requirements.txt`.

```bash
git clone https://github.com/zaza25266/customer_segementation.git
cd customer_segementation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
streamlit run app/streamlit_app.py
```

---

## Limitations

This is unsupervised learning — there are no ground-truth labels, so metrics measure cluster structure rather than prediction accuracy. A Silhouette Score of 0.3632 indicates moderate, not perfect, separation. Cluster labels are business interpretations, not known classes, and customer behavior can shift over time; a continuously operating system would need periodic retraining and monitoring.

---

## Future MLOps Improvements

MLflow tracking, DVC data versioning, Great Expectations / data validation, Evidently drift monitoring, automated retraining, GitHub Actions CI/CD, Docker, AWS and Kubernetes deployment, scheduled training, and model/data quality monitoring.

---

## Technologies

**ML:** Python, Pandas, NumPy, Scikit-learn, K-Means, Agglomerative Clustering, GMM, DBSCAN, PCA
**Production:** Joblib, Pytest, Hugging Face Hub, Streamlit
**Dev:** Git, GitHub, virtual environments

---

## Links

- **Live Demo:** https://customersegementation-t5s8rntqez9a2bdbahe2b8.streamlit.app/
- **GitHub:** https://github.com/zaza25266/customer_segementation
- **Hugging Face Model:** https://huggingface.co/ZubairAli25266/customer_segementation
- **GitHub:** https://github.com/zaza25266/customer_segementation
- **Hugging Face Model:** https://huggingface.co/ZubairAli25266/customer_segementation
