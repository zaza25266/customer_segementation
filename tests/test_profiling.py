import pandas as pd
import pytest

from src.customer_segmentation.evaluation.profiling import (
    profile_clusters,
    create_cluster_mapping,
)


def test_profile_clusters():

    customer_features = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4],
            "Recency": [300, 350, 20, 30],
            "Frequency": [1, 1, 8, 10],
            "Monetary": [300, 400, 2000, 2500],
            "Total_Quantity": [100, 120, 1000, 1200],
            "Unique_Products": [10, 15, 80, 100],
            "Average_Order_Value": [
                300,
                400,
                250,
                250,
            ],
        }
    )

    labels = [0, 0, 1, 1]

    profile = profile_clusters(
        customer_features,
        labels,
    )

    assert len(profile) == 2
    assert profile["Customer_Count"].sum() == 4


def test_cluster_mapping():

    profile = pd.DataFrame(
        {
            "Cluster": [0, 1],
            "Recency": [325, 25],
            "Monetary": [350, 2250],
        }
    )

    mapping = create_cluster_mapping(profile)

    assert (
        mapping[0]
        == "Inactive / Low-Value Customers"
    )

    assert (
        mapping[1]
        == "Active / High-Value Customers"
    )


def test_profile_clusters_rejects_mismatched_labels():

    customer_features = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "Recency": [30, 300],
            "Frequency": [5, 1],
            "Monetary": [1000, 200],
            "Total_Quantity": [500, 100],
            "Unique_Products": [50, 10],
            "Average_Order_Value": [200, 200],
        }
    )

    with pytest.raises(ValueError):

        profile_clusters(
            customer_features,
            [0],
        )