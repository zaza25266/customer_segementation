from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

SHEETS = [
    "Year 2009-2010",
    "Year 2010-2011",
]


def load_raw_data(file_path: str | Path) -> pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {file_path}"
        )

    dataframes = []

    for sheet_name in SHEETS:
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
            )
        except ValueError as exc:
            raise ValueError(
                f"Required sheet '{sheet_name}' was not found "
                f"in {file_path}."
            ) from exc

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Sheet '{sheet_name}' is missing columns: "
                f"{missing_columns}"
            )

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    return combined_df