import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture

def evaluate_kmeans_clusters(X_scaled, max_k=10):
    print("\n--- Evaluating KMeans ---")
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"KMeans | k={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def evaluate_birch_clusters(X_scaled, max_k=10):
    print("\n--- Evaluating Birch ---")
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        model = Birch(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Birch | k={k}, Silhouette Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def evaluate_gmm_clusters(X_scaled, max_k=10):
    print("\n--- Evaluating GMM ---")
    best_k = 2
    best_bic = np.inf
    for k in range(2, max_k + 1):
        model = GaussianMixture(n_components=k, random_state=42)
        model.fit(X_scaled)
        bic = model.bic(X_scaled)
        print(f"GMM | k={k}, BIC={bic:.4f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k

def main():
    file_path = "data/InSDN_Combined_Balanced_Shuffled.csv"
    df = pd.read_csv(file_path)

    # Drop non-numeric or irrelevant columns (adjust as needed)
    drop_cols = ['Label', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Protocol']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()  

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate clustering methods
    k_kmeans = evaluate_kmeans_clusters(X_scaled)
    k_birch = evaluate_birch_clusters(X_scaled)
    k_gmm = evaluate_gmm_clusters(X_scaled)

    print("\n--- Best Cluster Numbers ---")
    print(f"Best KMeans k: {k_kmeans}")
    print(f"Best Birch k: {k_birch}")
    print(f"Best GMM k: {k_gmm}")

if __name__ == "__main__":
    main()
