import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture

from dataset_loader.unsw_loader import load_unsw_datasets
from dataset_loader.ids2017_loader import load_ids2017_dataset
from dataset_loader.ids2018_loader import load_ids2018_dataset

from utils.hfc import hfc_pipeline, compute_feature_stats, suggest_hfc_parameters
from utils.pfe import pfe_pipeline
from utils.autoencoder import autoencoder_pipeline
from utils.mutual_info import mi_pipeline, mi_pipeline_transform
from utils.monitor import monitor_and_run

from models.random_forest import train_rf
from models.knn import train_knn
from models.mlp import train_mlp

def run_models(X_train, y_train, X_test, y_test, name=""):
    print(f"\n[{name} | Random Forest]")
    train_rf(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | KNN]")
    train_knn(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | MLP]")
    train_mlp(X_train, X_test, y_train, y_test)

def print_hfc_summary(clustering_name, chord_map, rule_map, coverage_map, total_samples):
    print(f"\n--- HFC Rule Summary ({clustering_name}) ---")
    for chord in chord_map:
        print(f"\n{chord} — Features: {chord_map[chord]}")
        print(f"Rule:\n{rule_map[chord]}")
        print(f"Coverage: {coverage_map[chord]} samples ({coverage_map[chord] / total_samples * 100:.2f}%)")

def run_hfc_with_clustering(X_train, y_train, X_test, y_test, clustering_method, clustering_name, dataset_type):
    print(f"\n--- Running HFC with {clustering_name} Clustering ---")
    (hfc_outputs_train, _) = monitor_and_run(hfc_pipeline, X_train, y_train, clustering_method=clustering_method)
    motifs_train_df, chord_map, rule_map, coverage_map = hfc_outputs_train
    print_hfc_summary(clustering_name, chord_map, rule_map, coverage_map, len(X_train))
    motifs_train_df['label'] = y_train.values

    expected_columns = motifs_train_df.drop(columns=["label"]).columns

    (hfc_outputs_test, _) = monitor_and_run(hfc_pipeline, X_test, y_test, clustering_method=clustering_method)
    motifs_test_df = hfc_outputs_test[0]
    motifs_test_df = motifs_test_df.reindex(columns=expected_columns, fill_value=0)
    motifs_test_df['label'] = y_test.values

    run_models(motifs_train_df.drop(columns=["label"]), y_train, motifs_test_df.drop(columns=["label"]), y_test, name=f"HFC-{dataset_type}-{clustering_name}")


def main():
    print("Choose dataset to use:")
    print("1. UNSW-NB15")
    print("2. CICIDS 2017 Combined")
    print("3. CSE-CIC-IDS2018 Downsampled")
    choice = input("Enter 1, 2, or 3: ")

    if choice == "1":
        train_path = "data/UNSW_NB15_training-set.csv"
        test_path = "data/UNSW_NB15_testing-set.csv"
        X_train, y_train, X_test, y_test = load_unsw_datasets(train_path, test_path)
        dataset_type = "UNSW"
    elif choice == "2":
        X_train, X_test, y_train, y_test = load_ids2017_dataset("data/CICIDS_2017_Combined_Balanced_Shuffled.csv")
        dataset_type = "CICIDS2017"
    elif choice == "3":
        X_train, X_test, y_train, y_test = load_ids2018_dataset("data/InSDN_Combined_Balanced_Shuffled.csv")
        dataset_type = "CSE-CIC-IDS2018"
    else:
        print("Invalid choice.")
        return

    # === HFC for all clustering methods
    run_hfc_with_clustering(X_train, y_train, X_test, y_test, KMeans(n_clusters=7), "KMeans", dataset_type)
    run_hfc_with_clustering(X_train, y_train, X_test, y_test, GaussianMixture(n_components=9), "GMM", dataset_type)
    run_hfc_with_clustering(X_train, y_train, X_test, y_test, Birch(n_clusters=2), "Birch", dataset_type)

    # === PFE
    # pfe_train, _ = monitor_and_run(pfe_pipeline, X_train)
    # pfe_train['label'] = y_train.values
    # pfe_test, _ = monitor_and_run(pfe_pipeline, X_test)
    # pfe_test['label'] = y_test.values
    # run_models(pfe_train.drop(columns=["label"]), y_train, pfe_test.drop(columns=["label"]), y_test, name="PFE")

    # === Mutual Information
    # (mi_train, mi_selector), _ = monitor_and_run(mi_pipeline, X_train, y_train)
    # mi_train['label'] = y_train.values
    # mi_test, _ = monitor_and_run(mi_pipeline_transform, X_test, mi_selector)
    # mi_test['label'] = y_test.values
    # run_models(mi_train.drop(columns=["label"]), y_train, mi_test.drop(columns=["label"]), y_test, name="MI")

    # === Autoencoder
    # ae_train, _ = monitor_and_run(autoencoder_pipeline, X_train)
    # ae_train['label'] = y_train.values
    # ae_test, _ = monitor_and_run(autoencoder_pipeline, X_test)
    # ae_test['label'] = y_test.values
    # run_models(ae_train.drop(columns=["label"]), y_train, ae_test.drop(columns=["label"]), y_test, name="Autoencoder")

if __name__ == "__main__":
    main()