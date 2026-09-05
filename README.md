# Production MLOps Pipeline

### End-to-End Clustering, Deployment, Monitoring, and Model Lifecycle System

## Project Overview

This project demonstrates a production-oriented MLOps lifecycle for an unsupervised machine learning system.

The machine learning workload is customer segmentation using clustering. The primary objective is not to develop a novel clustering algorithm, but to demonstrate how an ML system can be developed, versioned, evaluated, deployed, monitored, and retrained.

## Dataset

**UCI Online Retail II**

The dataset contains transaction records from a UK-based online retailer covering 2009–2011.

Source: UCI Machine Learning Repository

The raw transactional data will be transformed into customer-level behavioral features for clustering.

## Planned Architecture

```text
Raw Data
   ↓
DVC
   ↓
Pandera Validation
   ↓
Data Processing
   ↓
Customer Feature Engineering
   ↓
Clustering
   ↓
Evaluation
   ↓
MLflow
   ↓
Model Registry
   ↓
Champion Model
   ↓
FastAPI
   ↓
Docker
   ↓
CI/CD
   ↓
AWS EC2
   ↓
Monitoring
```

## Technology Stack

### Data

* Python
* Pandas
* NumPy
* Pandera
* DVC

### Machine Learning

* Scikit-learn
* KMeans

### Experiment Tracking

* MLflow

### API

* FastAPI
* Pydantic

### Database

* PostgreSQL

### Testing

* pytest

### Containerization

* Docker
* Docker Compose

### CI/CD

* GitHub Actions
* GitHub Container Registry

### Cloud

* AWS EC2
* Amazon RDS
* Amazon S3
* AWS IAM
* AWS Secrets Manager
* GitHub Actions OIDC

### Monitoring

* Prometheus
* Grafana
* Evidently

## Current Status

Project initialization.

The clustering methodology, feature engineering strategy, validation rules, model evaluation, deployment architecture, monitoring strategy, and retraining workflow will be developed incrementally.

## MLOps Lifecycle

The final system will demonstrate:

1. Data versioning
2. Data validation
3. Reproducible processing
4. Clustering model training
5. Unsupervised model evaluation
6. Experiment tracking
7. Model registry
8. Champion/candidate comparison
9. API deployment
10. Containerization
11. CI/CD
12. Production monitoring
13. Data drift detection
14. Retraining decisions
15. Model version promotion

## Project Structure

```text
production-mlops-pipeline/
├── api/
├── src/
│   ├── data/
│   ├── training/
│   └── monitoring/
├── tests/
├── notebooks/
├── monitoring/
├── deploy/
├── data/
├── configs/
├── .github/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md

