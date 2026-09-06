
import pandas as pd

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
  
    required_columns = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned_df = df.copy()

    # 1. Remove exact duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # 2. Remove cancellation invoices
    cancellation_mask = (
        cleaned_df["Invoice"]
        .astype(str)
        .str.startswith("C")
    )

    cleaned_df = cleaned_df.loc[~cancellation_mask]

    # 3. Keep positive quantities
    cleaned_df = cleaned_df.loc[
        cleaned_df["Quantity"] > 0
    ]

    # 4. Keep positive prices
    cleaned_df = cleaned_df.loc[
        cleaned_df["Price"] > 0
    ]

    # 5. Keep records with customer identifiers
    cleaned_df = cleaned_df.loc[
        cleaned_df["Customer ID"].notna()
    ]

    # 6. Keep UK customers
    cleaned_df = cleaned_df.loc[
        cleaned_df["Country"] == "United Kingdom"
    ]

    return cleaned_df.reset_index(drop=True)