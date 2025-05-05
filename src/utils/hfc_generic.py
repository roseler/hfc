from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def hfc_pipeline_fit(X, y, clustering_method=None, contrast_threshold=0.3, cov_threshold=1, min_votes=3):
    """
    Unified HFC pipeline for all datasets. Accepts any clustering method.
    Default clustering is KMeans(n_clusters=3).
    """

    from sklearn.cluster import KMeans

    # === Step 1: Scale features ===
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # === Step 2: Apply clustering ===
    if clustering_method is None:
        clustering_method = KMeans(n_clusters=3, random_state=42)

    clusters = clustering_method.fit_predict(X_scaled)
    clusters = pd.Series(clusters, index=X_scaled.index)

    # === Step 3: Global statistics for contrast calculation ===
    global_mean = X_scaled.mean()
    global_std = X_scaled.std()

    chord_map, rule_map, coverage_map, motif_list = {}, {}, {}, []

    for k in sorted(clusters.unique()):
        cluster_data = X_scaled[clusters == k]
        cluster_mean = cluster_data.mean()
        contrast_score = ((cluster_mean - global_mean).abs() / global_std).fillna(0)

        selected_features = contrast_score[contrast_score > contrast_threshold].index.tolist()
        if not selected_features:
            continue

        # Binary condition matrix per feature
        condition_matches = pd.DataFrame({
            f: (X_scaled[f] > cluster_mean[f]).astype(int)
            for f in selected_features
        })

        votes = condition_matches.sum(axis=1)
        motif_series = (votes >= min_votes).astype(int)
        chord_name = f"hfc_chord_c{k}"
        motif_series.name = chord_name
        motif_list.append(motif_series)

        chord_map[chord_name] = selected_features

        # Avoid closure overwrite by using factory function
        def make_rule(feats, means, threshold):
            return lambda df: (
                (df[feats] > means[feats]).astype(int).sum(axis=1) >= threshold
            ).astype(int)

        rule_map[chord_name] = make_rule(selected_features, cluster_mean.copy(), min_votes)
        coverage_map[chord_name] = int(motif_series.sum())

    motifs_df = pd.concat(motif_list, axis=1) if motif_list else pd.DataFrame(index=X.index)
    motifs_df['label'] = y.reset_index(drop=True)

    # === Summary ===
    print("\n[Chord Activation Summary by Class]")
    print(motifs_df.groupby('label').sum())
    print(f"Generated {len(motif_list)} HFC motif chords.")

    return motifs_df, chord_map, rule_map, coverage_map, scaler


def hfc_pipeline_transform(X, rule_map, scaler):
    """
    Applies pre-generated rule_map to a scaled dataset using the given scaler.
    """
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return pd.DataFrame({chord: rule(X_scaled) for chord, rule in rule_map.items()}, index=X.index)
