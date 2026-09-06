

from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split

from src.customer_segmentation.data.aggregator import (
    aggregate_customer_features,
)
from src.customer_segmentation.data.cleaner import clean_transactions
from src.customer_segmentation.data.loader import load_raw_data

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.customer_segmentation.models.pipeline import (
    build_customer_segmentation_pipeline,
)

from src.customer_segmentation.models.metadata import (
    save_model_metadata,
)

from src.customer_segmentation.evaluation.profiling import (
    profile_clusters,
    create_cluster_mapping,
)


RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
MODEL_PATH = Path(
    "models/customer_segmentation_pipeline_v2.joblib"
)
METADATA_PATH = Path(
    "models/customer_segmentation_metadata_v2.json"
)
FINAL_PREDICTIONS_PATH = Path(
    "data/processed/final_customer_segmentation.csv"
)


def prepare_customer_data(
    raw_data_path: str | Path = RAW_DATA_PATH,
):

    # 1. Load raw transactions
    raw_df = load_raw_data(raw_data_path)

    # 2. Clean transactions
    clean_df = clean_transactions(raw_df)

    # 3. Aggregate transactions into customer-level features
    customer_df = aggregate_customer_features(clean_df)

    # 4. Separate identifier from ML features
    X = customer_df.drop(columns=["Customer ID"])
    customer_ids = customer_df["Customer ID"]

    # 5. Create customer-level train/holdout split
    X_train, X_holdout, ids_train, ids_holdout = train_test_split(
        X,
        customer_ids,
        test_size=0.20,
        random_state=42,
    )

    return (
        X_train,
        X_holdout,
        ids_train,
        ids_holdout,
    )



def train_and_evaluate(
    X_train,
    X_holdout,
    ids_train,
):

    # Build the complete preprocessing + clustering pipeline.
    pipeline = build_customer_segmentation_pipeline()

    # Fit preprocessing and K-Means using training customers only.
    pipeline.fit(X_train)

        # Predict the same training customers used to fit K-Means.
    train_labels = pipeline.predict(X_train)

    # Reconstruct the training customer dataset with IDs
    # so cluster profiles can be interpreted in business terms.
    training_customers = X_train.copy()
    training_customers.insert(
        0,
        "Customer ID",
        ids_train.values,
    )

    cluster_profile = profile_clusters(
        training_customers,
        train_labels,
    )

    cluster_mapping = create_cluster_mapping(
        cluster_profile,
    )

    # Predict clusters for unseen holdout customers.
    holdout_labels = pipeline.predict(X_holdout)

    # Transform holdout features using the fitted preprocessing.
    X_holdout_transformed = pipeline.named_steps[
        "preprocessor"
    ].transform(X_holdout)

    # Calculate clustering metrics on holdout data.
    silhouette = silhouette_score(
        X_holdout_transformed,
        holdout_labels,
    )

    davies_bouldin = davies_bouldin_score(
        X_holdout_transformed,
        holdout_labels,
    )

    calinski_harabasz = calinski_harabasz_score(
        X_holdout_transformed,
        holdout_labels,
    )

    metrics = {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }

    return pipeline, metrics, cluster_mapping



def save_model(
    pipeline,
    model_path: str | Path = MODEL_PATH,
):
    
    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

def save_predictions(
    predictions,
    output_path: str | Path = FINAL_PREDICTIONS_PATH,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Predictions saved to: {output_path}"
    )

def generate_customer_predictions(
    pipeline,
    cluster_mapping,
    raw_data_path: str | Path = RAW_DATA_PATH,
):
    raw_df = load_raw_data(raw_data_path)

    clean_df = clean_transactions(raw_df)

    customer_df = aggregate_customer_features(
        clean_df
    )

    features = customer_df.drop(
        columns=["Customer ID"]
    )

    cluster_ids = pipeline.predict(features)

    predictions = customer_df.copy()

    predictions["Cluster"] = cluster_ids

    predictions["Segment"] = (
        predictions["Cluster"]
        .map(cluster_mapping)
    )

    return predictions



def create_model_metadata(
    metrics: dict,
    training_customers: int,
    holdout_customers: int,
    cluster_mapping: dict[int, str],
) -> dict:

    return {
        "model": "KMeans",
        "n_clusters": 2,
        "random_state": 42,
        "n_init": 10,
        "features": [
            "Recency",
            "Frequency",
            "Monetary",
            "Total_Quantity",
            "Unique_Products",
            "Average_Order_Value",
        ],
        "log_features": [
            "Frequency",
            "Monetary",
            "Total_Quantity",
            "Unique_Products",
            "Average_Order_Value",
        ],
        "training_customers": training_customers,
        "holdout_size": holdout_customers,
        "holdout_metrics": metrics,
        "cluster_interpretation": {
                str(cluster): label
                for cluster, label in cluster_mapping.items()
        },
    }


if __name__ == "__main__":

    (
        X_train,
        X_holdout,
        ids_train,
        ids_holdout,
    ) = prepare_customer_data()

    pipeline, metrics, cluster_mapping = train_and_evaluate(
        X_train,
        X_holdout,
        ids_train,
    )

    print("Training features:", X_train.shape)
    print("Holdout features:", X_holdout.shape)

    print("\nHoldout metrics:")
    print(f"Silhouette: {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    print(
        f"Calinski-Harabasz: "
        f"{metrics['calinski_harabasz']:.4f}"
    )

    print("\nCluster mapping:")
    print(cluster_mapping)

    save_model(pipeline)
    
    predictions = generate_customer_predictions(
    pipeline=pipeline,
    cluster_mapping=cluster_mapping,
)

    save_predictions(predictions)

    metadata = create_model_metadata(
        metrics=metrics,
        training_customers=len(ids_train),
        holdout_customers=len(ids_holdout),
        cluster_mapping=cluster_mapping,
    )

    save_model_metadata(
        metadata,
        METADATA_PATH,
    )