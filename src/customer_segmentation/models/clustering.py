from sklearn.cluster import KMeans


def build_kmeans_model(
    n_clusters: int = 2,
    random_state: int = 42,
    n_init: int = 10,
) -> KMeans:
 
    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )