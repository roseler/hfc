from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import pandas as pd
import numpy as np

def hfc_pipeline_fit_cse2018(X, y, contrast_threshold=0.3, cov_threshold=1.0, min_votes=2):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    clustering = Birch(n_clusters=3)
    clusters = clustering.fit_predict(X_scaled)
    clusters = pd.Series(clusters, index=X_scaled.index)

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

        condition_matches = pd.DataFrame({
            f: (X_scaled[f] > cluster_mean[f]).astype(int)
            for f in selected_features
        })

        votes = condition_matches.sum(axis=1)
        motif_series = (votes >= min_votes).astype(int)
        chord_name = f"cse2018_chord_c{k}"
        motif_series.name = chord_name
        motif_list.append(motif_series)

        chord_map[chord_name] = selected_features
        rule_map[chord_name] = lambda df, feats=selected_features, means=cluster_mean.copy(), threshold=min_votes: (
            (df[feats] > means[feats]).astype(int).sum(axis=1) >= threshold
        ).astype(int)
        coverage_map[chord_name] = int(motif_series.sum())

        motifs_df = pd.DataFrame({chord: rule(X_scaled) for chord, rule in rule_map.items()})
        motifs_df['label'] = y.reset_index(drop=True)

    print("\\n[Chord Activation Summary by Class]")
    print(motifs_df.groupby("label").sum())
    print(f"Generated {len(motif_list)} HFC motif chords for CSE2018.")

    return motifs_df, chord_map, rule_map, coverage_map, scaler

def hfc_pipeline_transform_cse2018(X, rule_map, scaler):
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return pd.DataFrame({chord: rule(X_scaled) for chord, rule in rule_map.items()})