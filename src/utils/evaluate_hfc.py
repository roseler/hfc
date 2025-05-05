import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture

from dataset_loader.unsw_loader import load_unsw_datasets
from dataset_loader.ids2017_loader import load_ids2017_dataset
from dataset_loader.inSDN_loader import load_inSDN_dataset

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
    (hfc_outputs_train, train_time) = monitor_and_run(
        hfc_pipeline, X_train, y_train, clustering_method=clustering_method
    )
    motifs_train_df, chord_map, rule_map, coverage_map = hfc_outputs_train
    print_hfc_summary(clustering_name, chord_map, rule_map, coverage_map, len(X_train))
    motifs_train_df['label'] = y_train.values

    expected_columns = motifs_train_df.drop(columns=["label"]).columns

    (hfc_outputs_test, test_time) = monitor_and_run(
        hfc_pipeline, X_test, y_test, clustering_method=clustering_method
    )
    motifs_test_df = hfc_outputs_test[0]
    motifs_test_df = motifs_test_df.reindex(columns=expected_columns, fill_value=0)
    motifs_test_df['label'] = y_test.values

    # Export CSVs
    export_dir = f"exports/{dataset_type}/HFC_{clustering_name}"
    os.makedirs(export_dir, exist_ok=True)
    motifs_train_df.to_csv(f"{export_dir}/hfc_train.csv", index=False)
    motifs_test_df.to_csv(f"{export_dir}/hfc_test.csv", index=False)

    run_models(
        motifs_train_df.drop(columns=["label"]),
        y_train,
        motifs_test_df.drop(columns=["label"]),
        y_test,
        name=f"HFC-{dataset_type}-{clustering_name}"
    )
    return train_time, test_time

def run_other_pipeline(X_train, y_train, X_test, y_test, pipeline_func, name, dataset_type):
    print(f"\n--- Running {name} Pipeline ---")
    # Training
    if name == "MI":
        (mi_train_tuple, train_time) = monitor_and_run(pipeline_func, X_train, y_train)
        mi_train_df, selector = mi_train_tuple
        mi_train_df['label'] = y_train.values
        # Testing
        (mi_test_df, test_time) = monitor_and_run(mi_pipeline_transform, X_test, selector)
        mi_test_df['label'] = y_test.values
        train_output, test_output = mi_train_df, mi_test_df
    else:
        (train_output, train_time) = monitor_and_run(pipeline_func, X_train)
        train_output['label'] = y_train.values
        (test_output, test_time) = monitor_and_run(pipeline_func, X_test)
        test_output['label'] = y_test.values

    # Export CSVs
    export_dir = f"exports/{dataset_type}/{name}"
    os.makedirs(export_dir, exist_ok=True)
    train_output.to_csv(f"{export_dir}/{name.lower()}_train.csv", index=False)
    test_output.to_csv(f"{export_dir}/{name.lower()}_test.csv", index=False)

    run_models(
        train_output.drop(columns=["label"]),
        y_train,
        test_output.drop(columns=["label"]),
        y_test,
        name=f"{name}-{dataset_type}"
    )
    return train_time, test_time

def main():
    print("Choose dataset to use:")
    print("1. UNSW-NB15")
    print("2. CICIDS 2017 Combined")
    print("3. InSDN")
    choice = input("Enter 1, 2, or 3: ")

    if choice == "1":
        X_train, y_train, X_test, y_test = load_unsw_datasets(
            "data/UNSW_NB15_training-set.csv",
            "data/UNSW_NB15_testing-set.csv"
        )
        dataset_type = "UNSW"
    elif choice == "2":
        X_train, X_test, y_train, y_test = load_ids2017_dataset(
            "data/CICIDS_2017_Combined_Balanced_Shuffled.csv"
        )
        dataset_type = "CICIDS2017"
    elif choice == "3":
        X_train, X_test, y_train, y_test = load_inSDN_dataset(
            "data/InSDN_Combined_Balanced_Shuffled.csv"
        )
        dataset_type = "InSDN"
    else:
        print("Invalid choice.")
        return

    runtimes = {}
    # HFC
    runtimes["HFC_KMeans"] = run_hfc_with_clustering(
        X_train, y_train, X_test, y_test,
        KMeans(n_clusters=2), "KMeans", dataset_type
    )
    runtimes["HFC_GMM"] = run_hfc_with_clustering(
        X_train, y_train, X_test, y_test,
        GaussianMixture(n_components=10), "GMM", dataset_type
    )
    runtimes["HFC_Birch"] = run_hfc_with_clustering(
        X_train, y_train, X_test, y_test,
        Birch(n_clusters=2), "Birch", dataset_type
    )

    # PFE
    runtimes["PFE"] = run_other_pipeline(
        X_train, y_train, X_test, y_test,
        pfe_pipeline, "PFE", dataset_type
    )

    # Mutual Information
    runtimes["MI"] = run_other_pipeline(
        X_train, y_train, X_test, y_test,
        mi_pipeline, "MI", dataset_type
    )

    # Autoencoder
    runtimes["Autoencoder"] = run_other_pipeline(
        X_train, y_train, X_test, y_test,
        autoencoder_pipeline, "Autoencoder", dataset_type
    )

    # Print all runtimes
    print("\n=== Monitor Results (Training and Testing Runtime in seconds) ===")
    for method, (train_t, test_t) in runtimes.items():
        print(f"{method} - Train: {train_t:.2f}s, Test: {test_t:.2f}s")

if __name__ == "__main__":
    main()
