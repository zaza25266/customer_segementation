
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json

import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from src.customer_segmentation.features.transformer import (
    FEATURE_COLUMNS,
)
from src.customer_segmentation.models.pipeline import (
    build_customer_segmentation_pipeline,
)


HF_REPO_ID = "ZubairAli25266/customer_segementation"

MODEL_FILENAME = (
    "customer_segmentation_pipeline_v2.joblib"
)

METADATA_FILENAME = (
    "customer_segmentation_metadata_v2.json"
)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
    )

    import joblib

    return joblib.load(model_path)


@st.cache_data
def load_metadata():
    metadata_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=METADATA_FILENAME,
    )

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_input(features: dict) -> None:
    for feature, value in features.items():
        if value < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

    if features["Frequency"] < 1:
        raise ValueError(
            "Frequency must be at least 1."
        )

    if features["Unique_Products"] < 1:
        raise ValueError(
            "Unique Products must be at least 1."
        )


def predict_segment(
    model,
    metadata: dict,
    features: dict,
):
    validate_input(features)

    input_df = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS,
    )

    cluster_id = int(
        model.predict(input_df)[0]
    )

    cluster_mapping = {
        int(cluster): label
        for cluster, label
        in metadata["cluster_interpretation"].items()
    }

    segment = cluster_mapping.get(
        cluster_id,
        "Unknown Segment",
    )

    return cluster_id, segment


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)


st.title("Customer Segmentation")
st.markdown(
    """
    ### RFM-based Customer Segmentation

    Predict whether a customer belongs to the
    **Active / High-Value** or
    **Inactive / Low-Value** segment using
    behavioral purchasing features.
    """
)


try:
    model = load_model()
    metadata = load_metadata()

except Exception as exc:
    st.error(
        "Failed to load the model from Hugging Face."
    )
    st.exception(exc)
    st.stop()


st.sidebar.header("Model Information")

st.sidebar.write(
    f"**Model:** {metadata.get('model', 'Unknown')}"
)

st.sidebar.write(
    f"**Clusters:** "
    f"{metadata.get('n_clusters', 'Unknown')}"
)

st.sidebar.write(
    f"**Training Customers:** "
    f"{metadata.get('training_customers', 'Unknown')}"
)

st.sidebar.write(
    "**Features:**"
)

for feature in metadata.get("features", []):
    st.sidebar.write(f"- {feature}")


st.header("Customer Information")


col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=30,
        step=1,
    )

    frequency = st.number_input(
        "Frequency (orders)",
        min_value=1,
        value=5,
        step=1,
    )


with col2:
    monetary = st.number_input(
        "Monetary (£)",
        min_value=0.0,
        value=1500.0,
        step=50.0,
    )

    total_quantity = st.number_input(
        "Total Quantity",
        min_value=1,
        value=500,
        step=10,
    )


with col3:
    unique_products = st.number_input(
        "Unique Products",
        min_value=1,
        value=50,
        step=1,
    )

    average_order_value = st.number_input(
        "Average Order Value (£)",
        min_value=0.0,
        value=300.0,
        step=10.0,
    )


features = {
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary,
    "Total_Quantity": total_quantity,
    "Unique_Products": unique_products,
    "Average_Order_Value": average_order_value,
}


if st.button(
    "Predict Customer Segment",
    type="primary",
    use_container_width=True,
):

    try:

        cluster_id, segment = predict_segment(
            model=model,
            metadata=metadata,
            features=features,
        )

        st.divider()

        st.subheader("Prediction")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Cluster",
                cluster_id,
            )

        with result_col2:
            st.metric(
                "Customer Segment",
                segment,
            )

        st.success(
            f"Customer belongs to: **{segment}**"
        )

        st.subheader("Input Features")

        input_display = pd.DataFrame(
            {
                "Feature": list(features.keys()),
                "Value": list(features.values()),
            }
        )

        st.dataframe(
            input_display,
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            "Prediction failed."
        )
        st.exception(exc)


st.divider()

st.caption(
    "Model trained on the Online Retail II dataset. "
    "Model artifacts are loaded from Hugging Face."
)
