import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from src.customer_segmentation.data.cleaner import (
    clean_transactions,
)

from src.customer_segmentation.data.aggregator import (
    aggregate_customer_features,
)


# Configuration ---------------------------------------

HF_REPO_ID = "ZubairAli25266/customer_segementation"

MODEL_FILENAME = (
    "customer_segmentation_pipeline_v2.joblib"
)

METADATA_FILENAME = (
    "customer_segmentation_metadata_v2.json"
)

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


# Model loading ---------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
    )

    return joblib.load(model_path)


@st.cache_data
def load_metadata():

    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=METADATA_FILENAME,
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Data validation -------------------------------------

def validate_raw_data(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Uploaded file is missing required "
            f"columns: {missing_columns}"
        )


# Feature engineering ---------------------------------

def prepare_customer_features(
    raw_data: pd.DataFrame,
):

    validate_raw_data(raw_data)

    cleaned_data = clean_transactions(
        raw_data
    )

    if cleaned_data.empty:

        raise ValueError(
            "No valid transactions remain after "
            "cleaning."
        )

    customer_features = (
        aggregate_customer_features(
            cleaned_data
        )
    )

    if customer_features.empty:

        raise ValueError(
            "No customer records were created "
            "from the uploaded data."
        )

    return (
        cleaned_data,
        customer_features,
    )


# Prediction ------------------------------------------

def predict_customers(
    model,
    metadata: dict,
    customer_features: pd.DataFrame,
):

    feature_columns = metadata["features"]

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in customer_features.columns
    ]

    if missing_features:

        raise ValueError(
            "Required model features were not "
            f"generated: {missing_features}"
        )

    prediction_input = (
        customer_features[
            feature_columns
        ].copy()
    )

    cluster_ids = model.predict(
        prediction_input
    )

    predictions = customer_features.copy()

    predictions["Cluster"] = cluster_ids

    cluster_mapping = {
        int(cluster): label
        for cluster, label
        in metadata[
            "cluster_interpretation"
        ].items()
    }

    predictions["Segment"] = (
        predictions["Cluster"]
        .map(cluster_mapping)
    )

    return predictions


# Page configuration ----------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)


# Page header -----------------------------------------

st.title(
    "Customer Segmentation"
)

st.markdown(
    """
    ### Customer Segmentation from Transaction Data

    Upload raw transaction-level data using the
    expected company data format.

    The application automatically performs:

    **Raw Transactions → Cleaning → Customer Aggregation
    → Feature Engineering → Model Prediction → Segment**
    """
)


# Load model and metadata -----------------------------

try:

    model = load_model()

    metadata = load_metadata()

except Exception as exc:

    st.error(
        "Failed to load the model from Hugging Face."
    )

    st.exception(exc)

    st.stop()


# Sidebar ---------------------------------------------

st.sidebar.header(
    "Model Information"
)

st.sidebar.write(
    f"**Model:** "
    f"{metadata.get('model', 'Unknown')}"
)

st.sidebar.write(
    f"**Clusters:** "
    f"{metadata.get('n_clusters', 'Unknown')}"
)

st.sidebar.write(
    f"**Training Customers:** "
    f"{metadata.get('training_customers', 'Unknown')}"
)

st.sidebar.divider()

st.sidebar.write(
    "**Expected Raw Columns:**"
)

for column in EXPECTED_COLUMNS:

    st.sidebar.write(
        f"- {column}"
    )

st.sidebar.divider()

st.sidebar.write(
    "**Generated Model Features:**"
)

for feature in metadata.get(
    "features",
    [],
):

    st.sidebar.write(
        f"- {feature}"
    )


# File upload -----------------------------------------

st.header(
    "Upload Transaction Data"
)

st.write(
    "Upload a CSV containing raw transaction-level "
    "data. Customer-level features will be generated "
    "automatically."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"],
)


# Process uploaded data -------------------------------

if uploaded_file is not None:

    try:

        raw_data = pd.read_csv(
            uploaded_file
        )

        validate_raw_data(
            raw_data
        )

        # Raw data preview ---------------------------

        st.subheader(
            "Uploaded Data"
        )

        raw_col1, raw_col2 = st.columns(2)

        with raw_col1:

            st.metric(
                "Raw Transactions",
                f"{len(raw_data):,}",
            )

        with raw_col2:

            st.metric(
                "Raw Columns",
                f"{len(raw_data.columns):,}",
            )

        with st.expander(
            "Preview Raw Transaction Data"
        ):

            st.dataframe(
                raw_data.head(20),
                hide_index=True,
                use_container_width=True,
            )


        # Cleaning and feature engineering -----------

        with st.spinner(
            "Cleaning transactions and generating "
            "customer features..."
        ):

            (
                cleaned_data,
                customer_features,
            ) = prepare_customer_features(
                raw_data
            )


        st.success(
            "Transaction preprocessing completed."
        )


        # Data summary -------------------------------

        st.subheader(
            "Data Summary"
        )

        summary_col1, summary_col2 = (
            st.columns(2)
        )

        with summary_col1:

            st.metric(
                "Valid Transactions",
                f"{len(cleaned_data):,}",
            )

        with summary_col2:

            st.metric(
                "Customers",
                f"{len(customer_features):,}",
            )


        # Prediction ---------------------------------

        if st.button(
            "Predict Customer Segments",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Predicting customer segments..."
            ):

                predictions = predict_customers(
                    model=model,
                    metadata=metadata,
                    customer_features=(
                        customer_features
                    ),
                )


            st.success(
                "Customer segmentation completed."
            )


            # Segment distribution ------------------

            st.subheader(
                "Segment Distribution"
            )

            segment_counts = (
                predictions[
                    "Segment"
                ]
                .value_counts()
                .rename(
                    "Customers"
                )
                .reset_index()
            )

            st.dataframe(
                segment_counts,
                hide_index=True,
                use_container_width=True,
            )


            # Generated features --------------------

            st.subheader(
                "Generated Customer Features"
            )

            st.write(
                "These features were automatically "
                "created from the uploaded raw "
                "transaction data and used by the "
                "model for prediction."
            )

            generated_feature_columns = [
                "Customer ID",
                "Recency",
                "Frequency",
                "Monetary",
                "Total_Quantity",
                "Unique_Products",
                "Average_Order_Value",
            ]

            st.dataframe(
                predictions[
                    generated_feature_columns
                ],
                hide_index=True,
                use_container_width=True,
            )


            # Prediction results --------------------

            st.subheader(
                "Prediction Results"
            )

            result_columns = [
                "Customer ID",
                "Recency",
                "Frequency",
                "Monetary",
                "Total_Quantity",
                "Unique_Products",
                "Average_Order_Value",
                "Cluster",
                "Segment",
            ]

            st.dataframe(
                predictions[
                    result_columns
                ],
                hide_index=True,
                use_container_width=True,
            )


            # Segment metrics -----------------------

            st.subheader(
                "Business Segments"
            )

            active_count = int(
                (
                    predictions["Segment"]
                    == "Active / High-Value Customers"
                ).sum()
            )

            inactive_count = int(
                (
                    predictions["Segment"]
                    == "Inactive / Low-Value Customers"
                ).sum()
            )

            segment_col1, segment_col2 = (
                st.columns(2)
            )

            with segment_col1:

                st.metric(
                    "Active / High-Value Customers",
                    f"{active_count:,}",
                )

                st.write(
                    "Customers with stronger recent "
                    "activity and higher purchasing "
                    "value."
                )

            with segment_col2:

                st.metric(
                    "Inactive / Low-Value Customers",
                    f"{inactive_count:,}",
                )

                st.write(
                    "Customers with lower purchasing "
                    "activity and value."
                )


            # Download results ----------------------

            st.subheader(
                "Download Results"
            )

            csv_data = (
                predictions.to_csv(
                    index=False
                )
                .encode("utf-8")
            )

            st.download_button(
                label=(
                    "Download Segmentation Results"
                ),
                data=csv_data,
                file_name=(
                    "customer_segmentation_results.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


    except Exception as exc:

        st.error(
            "Failed to process the uploaded data."
        )

        st.exception(exc)


else:

    st.info(
        "Upload a CSV file containing raw "
        "transaction data to begin."
    )


# Footer ----------------------------------------------

st.divider()

st.caption(
    "Model trained on the Online Retail II dataset. "
    "Model and metadata are loaded from Hugging Face."
)
