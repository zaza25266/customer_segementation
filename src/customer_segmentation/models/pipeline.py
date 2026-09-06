
from sklearn.pipeline import Pipeline

from src.customer_segmentation.features.transformer import (
    build_feature_transformer,
)
from src.customer_segmentation.models.clustering import (
    build_kmeans_model,
)


def build_customer_segmentation_pipeline(
    n_clusters: int = 2,
    random_state: int = 42,
    n_init: int = 10,
) -> Pipeline:


    feature_transformer = build_feature_transformer()

    kmeans_model = build_kmeans_model(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", feature_transformer),
            ("clusterer", kmeans_model),
        ]
    )

    return pipeline