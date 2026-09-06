
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Average_Order_Value",
]

LOG_FEATURES = [
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Average_Order_Value",
]

RECENCY_FEATURE = ["Recency"]

def build_feature_transformer() -> ColumnTransformer:

    log_pipeline = Pipeline(
        steps=[
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "log_features",
                log_pipeline,
                LOG_FEATURES,
            ),
            (
                "recency",
                "passthrough",
                RECENCY_FEATURE,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_transformer", preprocessor),
            ("scaler", StandardScaler()),
        ]
    )