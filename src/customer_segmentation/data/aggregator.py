import pandas as pd


CUSTOMER_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Average_Order_Value",
]


def aggregate_customer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = {
        "Customer ID",
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    customer_df = df.copy()

    # Ensure dates are proper datetime values.
    customer_df["InvoiceDate"] = pd.to_datetime(
        customer_df["InvoiceDate"],
        errors="raise",
    )

    # Revenue for each transaction line.
    customer_df["Revenue"] = (
        customer_df["Quantity"] * customer_df["Price"]
    )

    # Reference date used for Recency.
    reference_date = (
        customer_df["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    # Customer-level aggregation.
    customer_features = (
        customer_df
        .groupby("Customer ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda dates: (
                    reference_date - dates.max()
                ).days,
            ),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            Total_Quantity=("Quantity", "sum"),
            Unique_Products=("StockCode", "nunique"),
        )
        .reset_index()
    )

    # Average order value.
    customer_features["Average_Order_Value"] = (
        customer_features["Monetary"]
        / customer_features["Frequency"]
    )

    return customer_features