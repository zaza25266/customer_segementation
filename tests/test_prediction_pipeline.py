import json

import pandas as pd
import pytest

from pipelines.prediction_pipeline import (
    validate_features,
    load_model_metadata,
)


def test_validate_features_accepts_valid_features():

    customer_features = pd.DataFrame(
        {
            "Recency": [30],
            "Frequency": [5],
            "Monetary": [1500.0],
            "Total_Quantity": [500],
            "Unique_Products": [50],
            "Average_Order_Value": [300.0],
        }
    )

    validate_features(customer_features)


def test_validate_features_rejects_missing_features():

    customer_features = pd.DataFrame(
        {
            "Recency": [30],
            "Frequency": [5],
            "Monetary": [1500.0],
        }
    )

    with pytest.raises(ValueError):
        validate_features(customer_features)


def test_validate_features_rejects_missing_values():

    customer_features = pd.DataFrame(
        {
            "Recency": [30],
            "Frequency": [5],
            "Monetary": [None],
            "Total_Quantity": [500],
            "Unique_Products": [50],
            "Average_Order_Value": [300.0],
        }
    )

    with pytest.raises(ValueError):
        validate_features(customer_features)


def test_load_model_metadata(tmp_path):

    metadata = {
        "model": "KMeans",
        "n_clusters": 2,
        "cluster_interpretation": {
            "0": "Active / High-Value Customers",
            "1": "Inactive / Low-Value Customers",
        },
    }

    metadata_path = tmp_path / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    loaded = load_model_metadata(
        metadata_path
    )

    assert loaded["model"] == "KMeans"
    assert loaded["n_clusters"] == 2
    assert (
        loaded["cluster_interpretation"]["0"]
        == "Active / High-Value Customers"
    )