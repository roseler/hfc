import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def hfc_pipeline(X, y=None, clustering_method=None, contrast_threshold=0.5):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Default to KMeans if no clustering method is passed
    if clustering_method is None:
        clustering_method = KMeans(n_clusters=2, random_state=42)

    # Fit clustering algorithm
    clusters = clustering_method.fit_predict(X_scaled)

    global_mean = X_scaled.mean()
    global_std = X_scaled.std()
    motif_list = []

    for k in sorted(pd.Series(clusters).unique()):
        cluster_data = X_scaled[pd.Series(clusters) == k]
        cluster_mean = cluster_data.mean()
        contrast_score = ((cluster_mean - global_mean).abs() / global_std).fillna(0)
        selected_features = contrast_score[contrast_score > contrast_threshold].index.tolist()

        if selected_features:
            motif_feature = pd.Series(1, index=X_scaled.index)
            for feature in selected_features:
                motif_feature &= (X_scaled[feature] > cluster_mean[feature]).astype(int)
            motif_feature.name = f'hfc_chord_c{k}'
            motif_list.append(motif_feature)

    if motif_list:
        motifs_df = pd.concat(motif_list, axis=1)
        final_df = pd.concat([X_scaled, motifs_df], axis=1)
    else:
        final_df = X_scaled.copy()

    if y is not None:
        final_df['label'] = y.values

    return final_df