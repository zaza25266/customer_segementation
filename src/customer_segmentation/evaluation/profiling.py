

import pandas as pd


PROFILE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Average_Order_Value",
]


def profile_clusters(
    customer_features: pd.DataFrame,
    cluster_labels,
) -> pd.DataFrame:

    required_columns = {
        "Customer ID",
        *PROFILE_COLUMNS,
    }

    missing_columns = required_columns - set(
        customer_features.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(customer_features) != len(cluster_labels):
        raise ValueError(
            "Number of customer records must match "
            "number of cluster labels."
        )

    df = customer_features.copy()

    df["Cluster"] = cluster_labels

    profile = (
        df.groupby("Cluster")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Recency=("Recency", "median"),
            Frequency=("Frequency", "median"),
            Monetary=("Monetary", "median"),
            Total_Quantity=("Total_Quantity", "median"),
            Unique_Products=("Unique_Products", "median"),
            Average_Order_Value=(
                "Average_Order_Value",
                "median",
            ),
            Total_Revenue=("Monetary", "sum"),
        )
        .reset_index()
    )

    total_revenue = profile["Total_Revenue"].sum()

    profile["Revenue_Share"] = (
        profile["Total_Revenue"]
        / total_revenue
    )

    return profile



def create_cluster_mapping(
    cluster_profile: pd.DataFrame,
) -> dict[int, str]:

    required_columns = {
        "Cluster",
        "Recency",
        "Monetary",
    }

    missing_columns = required_columns - set(
        cluster_profile.columns
    )

    if missing_columns:
        raise ValueError(
            f"Cluster profile is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Rank clusters by normalized business signals.
    profile = cluster_profile.copy()

    profile["business_score"] = (
        profile["Monetary"].rank(pct=True)
        + (-profile["Recency"]).rank(pct=True)
    )

    active_cluster = int(
        profile.loc[
            profile["business_score"].idxmax(),
            "Cluster",
        ]
    )

    inactive_cluster = int(
        profile.loc[
            profile["business_score"].idxmin(),
            "Cluster",
        ]
    )

    return {
        inactive_cluster: "Inactive / Low-Value Customers",
        active_cluster: "Active / High-Value Customers",
    }