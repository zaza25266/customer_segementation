from pathlib import Path

import joblib
import json
import pandas as pd


MODEL_PATH = Path(
    "models/customer_segmentation_pipeline_v2.joblib"
)

METADATA_PATH = Path(
    "models/customer_segmentation_metadata_v2.json"
)

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Average_Order_Value",
]


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def load_model_metadata(
    metadata_path: str | Path = METADATA_PATH,
) -> dict:
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_features(
    customer_features: pd.DataFrame,
) -> None:
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in customer_features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if customer_features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )


def predict_customer_segment(
    customer_features: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> pd.DataFrame:

    validate_features(customer_features)

    model = load_model(model_path)
    metadata = load_model_metadata(metadata_path)

    cluster_mapping = {
        int(cluster): label
        for cluster, label
        in metadata["cluster_interpretation"].items()
    }

    features = customer_features[
        FEATURE_COLUMNS
    ].copy()

    cluster_ids = model.predict(features)

    predictions = customer_features.copy()

    predictions["Cluster"] = cluster_ids

    predictions["Segment"] = predictions["Cluster"].map(
        cluster_mapping
    )

    return predictions


def predict_all_customers(
    customer_features: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
) -> pd.DataFrame:

    if "Customer ID" not in customer_features.columns:
        raise ValueError(
            "Customer features must contain 'Customer ID'."
        )

    return predict_customer_segment(
        customer_features=customer_features,
        model_path=model_path,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":

    from src.customer_segmentation.data.loader import load_raw_data
    from src.customer_segmentation.data.cleaner import clean_transactions
    from src.customer_segmentation.data.aggregator import (
        aggregate_customer_features,
    )

    raw_df = load_raw_data(
        "data/raw/online_retail_II.xlsx"
    )

    clean_df = clean_transactions(raw_df)

    customer_features = aggregate_customer_features(
        clean_df
    )

    predictions = predict_all_customers(
        customer_features
    )

    print("Customers:", len(predictions))

    print("\nSegment distribution:")
    print(
        predictions["Segment"]
        .value_counts()
    )

    print("\nSample predictions:")
    print(
        predictions[
            ["Customer ID", "Cluster", "Segment"]
        ].head(10)
    )

    print("\nCustomer 12747:")
    print(
        predictions.loc[
            predictions["Customer ID"] == 12747,
            ["Customer ID", "Cluster", "Segment"],
        ]
    )