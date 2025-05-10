import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import variation, zscore

def compute_feature_stats(X_scaled: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "std_dev": X_scaled.std(),
        "mean": X_scaled.mean(),
        "coefficient_of_variation": variation(X_scaled, axis=0),
        "zscore_mean": zscore(X_scaled, axis=0).mean(axis=0)
    })

def suggest_hfc_parameters(stats_df: pd.DataFrame, cov_threshold: float = 1.0):
    top_n = (stats_df['coefficient_of_variation'] > cov_threshold).sum()
    min_votes = max(2, int(np.ceil(0.5 * top_n)))
    return top_n, min_votes

def hfc_pipeline(X, y=None, clustering_method=None, contrast_threshold=0.5, cov_threshold=1, min_votes=None):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if clustering_method is None:
        clustering_method = KMeans(n_clusters=5, random_state=42)

    clusters = clustering_method.fit_predict(X_scaled)
    clusters = pd.Series(clusters, index=X_scaled.index)

    global_mean = X_scaled.mean()
    global_std = X_scaled.std()

    motif_list = []
    chord_feature_map = {}
    chord_rule_map = {}
    chord_coverage = {}

    stats_df = compute_feature_stats(X_scaled)
    auto_top_n, auto_min_votes = suggest_hfc_parameters(stats_df, cov_threshold)
    final_min_votes = min_votes if min_votes is not None else auto_min_votes

    for k in sorted(clusters.unique()):
        cluster_data = X_scaled[clusters == k]
        cluster_mean = cluster_data.mean()
        contrast_score = ((cluster_mean - global_mean).abs() / global_std).fillna(0)

        selected_features = contrast_score[contrast_score > contrast_threshold].index.tolist()
        if not selected_features:
            continue

        condition_matches = pd.DataFrame({
            f: (X_scaled[f] > cluster_mean[f]).astype(int)
            for f in selected_features
        })

        votes = condition_matches.sum(axis=1)
        motif_series = (votes >= final_min_votes).astype(int)
        chord_name = f"hfc_chord_c{k}"
        motif_series.name = chord_name
        motif_list.append(motif_series)

        chord_feature_map[chord_name] = selected_features
        chord_rule_map[chord_name] = (
            f"Match if ≥ {final_min_votes} of:\n" +
            "\n".join([f"{f} > {cluster_mean[f]:.3f}" for f in selected_features])
        )
        chord_coverage[chord_name] = int(motif_series.sum())

    motifs_df = pd.concat(motif_list, axis=1) if motif_list else pd.DataFrame(index=X.index)
    final_df = pd.concat([X_scaled, motifs_df], axis=1)

    if y is not None:
        final_df['label'] = y.values

    return final_df, chord_feature_map, chord_rule_map, chord_coverage
