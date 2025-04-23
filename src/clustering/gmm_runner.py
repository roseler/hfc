import pandas as pd
from sklearn.mixture import GaussianMixture
from utils.hfc import hfc_pipeline
from models.random_forest import train_rf
from models.knn import train_knn
from models.mlp import train_mlp

def run_gmm_hfc(X_train, y_train, X_test, y_test, min_votes):
    print("\n--- Running HFC with Gaussian Mixture Model (GMM) Clustering ---")

    motifs_train_df, chord_map, rule_map, coverage_map = hfc_pipeline(
        X_train, y_train,
        clustering_method=GaussianMixture(n_components=9, random_state=42),
        contrast_threshold=0.5,
        cov_threshold=1.0,
        min_votes=min_votes
    )

    print("\n--- HFC Rule Summary (GMM) ---")
    for chord in chord_map:
        print(f"\n{chord} — Features: {chord_map[chord]}")
        print(f"Rule:\n{rule_map[chord]}")
        print(f"Coverage: {coverage_map[chord]} samples ({coverage_map[chord]/len(X_train)*100:.2f}%)")

    motifs_test_df, _, _, _ = hfc_pipeline(
        X_test, y_test,
        clustering_method=GaussianMixture(n_components=9, random_state=42),
        contrast_threshold=0.5,
        cov_threshold=1.0,
        min_votes=min_votes
    )

    train_rf(motifs_train_df.drop(columns=["label"]), motifs_test_df.drop(columns=["label"]), y_train, y_test)
    train_knn(motifs_train_df.drop(columns=["label"]), motifs_test_df.drop(columns=["label"]), y_train, y_test)
    train_mlp(motifs_train_df.drop(columns=["label"]), motifs_test_df.drop(columns=["label"]), y_train, y_test)
