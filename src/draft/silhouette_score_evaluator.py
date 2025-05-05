# silhouette_score_evaluator.py

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def evaluate_k_by_silhouette(X, k_range=range(2, 11)):
    """
    Evaluate silhouette scores for a range of k values using KMeans clustering.

    Args:
        X (pd.DataFrame or np.array): Feature matrix.
        k_range (iterable): Range of cluster counts to test.

    Returns:
        best_k (int): The value of k with the highest silhouette score.
        scores (list of tuples): Each tuple contains (k, silhouette_score).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append((k, score))

    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k, scores
