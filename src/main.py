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

from utils.hfc import hfc_pipeline
from utils.pfe import pfe_pipeline
from utils.autoencoder import autoencoder_pipeline
from utils.mutual_info import mi_pipeline, mi_pipeline_transform
from utils.monitor import monitor_and_run

from models.random_forest import train_rf
from models.knn import train_knn
from models.mlp import train_mlp

# Import your new visualization module
from utils.visualization import visualize_results

def run_models(X_train, y_train, X_test, y_test, name=""):
    """Train and evaluate classifiers; return performance metrics if desired."""
    print(f"\n[{name} | Random Forest]")
    rf_metrics = train_rf(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | KNN]")
    knn_metrics = train_knn(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | MLP]")
    mlp_metrics = train_mlp(X_train, X_test, y_train, y_test)

    # Combine into a single dict for this run
    return {
        'RF': rf_metrics,
        'KNN': knn_metrics,
        'MLP': mlp_metrics
    }

def print_hfc_summary(clustering_name, chord_map, rule_map, coverage_map, total_samples):
    print(f"\n--- HFC Rule Summary ({clustering_name}) ---")
    for chord in chord_map:
        print(f"\n{chord} — Features: {chord_map[chord]}")
        print(f"Rule:\n{rule_map[chord]}")
        print(f"Coverage: {coverage_map[chord]} samples ({coverage_map[chord] / total_samples * 100:.2f}%)")

def run_hfc_with_clustering(X_train, y_train, X_test, y_test, clustering_method, clustering_name, dataset_type, results):
    """Run HFC, export CSVs, train models, and record metrics."""
    print(f"\n--- Running HFC with {clustering_name} Clustering ---")
    # HFC transformation + timing
    (hfc_train_outputs, train_time) = monitor_and_run(
        hfc_pipeline, X_train, y_train, clustering_method=clustering_method
    )
    motifs_train_df, chord_map, rule_map, coverage_map = hfc_train_outputs
    print_hfc_summary(clustering_name, chord_map, rule_map, coverage_map, len(X_train))
    motifs_train_df['label'] = y_train.values

    # Record HFC train runtime
    results['HFC'][clustering_name] = {
        'train_time': train_time,
        'chord_map': chord_map,
        'coverage_map': coverage_map
    }

    # Prepare test set
    expected_cols = motifs_train_df.drop(columns=['label']).columns
    (hfc_test_outputs, test_time) = monitor_and_run(
        hfc_pipeline, X_test, y_test, clustering_method=clustering_method
    )
    motifs_test_df = hfc_test_outputs[0].reindex(columns=expected_cols, fill_value=0)
    motifs_test_df['label'] = y_test.values

    # Record HFC test runtime
    results['HFC'][clustering_name]['test_time'] = test_time

    # Export CSVs
    export_dir = f"exports/{dataset_type}/HFC_{clustering_name}"
    os.makedirs(export_dir, exist_ok=True)
    motifs_train_df.to_csv(f"{export_dir}/hfc_train.csv", index=False)
    motifs_test_df.to_csv(f"{export_dir}/hfc_test.csv", index=False)

    # Train & evaluate classifiers
    model_metrics = run_models(
        motifs_train_df.drop(columns=['label']),
        y_train,
        motifs_test_df.drop(columns=['label']),
        y_test,
        name=f"HFC-{dataset_type}-{clustering_name}"
    )
    results['HFC'][clustering_name]['model_metrics'] = model_metrics

def run_other_pipeline(X_train, y_train, X_test, y_test, pipeline_func, name, dataset_type, results):
    """Run a generic pipeline (PFE, MI, AE), export CSVs, train models, and record metrics."""
    print(f"\n--- Running {name} Pipeline ---")
    # Train-time transform
    if name == "MI":
        (mi_train_tuple, train_time) = monitor_and_run(pipeline_func, X_train, y_train)
        mi_train_df, selector = mi_train_tuple
        mi_train_df['label'] = y_train.values

        # Test-time transform
        (mi_test_df, test_time) = monitor_and_run(mi_pipeline_transform, X_test, selector)
        mi_test_df['label'] = y_test.values

        train_df, test_df = mi_train_df, mi_test_df
    else:
        (train_df, train_time) = monitor_and_run(pipeline_func, X_train)
        train_df['label'] = y_train.values
        (test_df, test_time) = monitor_and_run(pipeline_func, X_test)
        test_df['label'] = y_test.values

    # Record runtimes
    results[name] = {
        'train_time': train_time,
        'test_time': test_time
    }

    # Export CSVs
    export_dir = f"exports/{dataset_type}/{name}"
    os.makedirs(export_dir, exist_ok=True)
    train_df.to_csv(f"{export_dir}/{name.lower()}_train.csv", index=False)
    test_df.to_csv(f"{export_dir}/{name.lower()}_test.csv", index=False)

    # Train & evaluate classifiers
    model_metrics = run_models(
        train_df.drop(columns=['label']),
        y_train,
        test_df.drop(columns=['label']),
        y_test,
        name=f"{name}-{dataset_type}"
    )
    results[name]['model_metrics'] = model_metrics

def main():
    # Initialize the results dictionary
    results = {
        'HFC': {},
        'PFE': {},
        'MI': {},
        'Autoencoder': {}
    }

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

    # Run HFC variants
    run_hfc_with_clustering(X_train, y_train, X_test, y_test,
                             KMeans(n_clusters=2), "KMeans", dataset_type, results)
    run_hfc_with_clustering(X_train, y_train, X_test, y_test,
                             GaussianMixture(n_components=10), "GMM", dataset_type, results)
    run_hfc_with_clustering(X_train, y_train, X_test, y_test,
                             Birch(n_clusters=2), "Birch", dataset_type, results)

    # Run other pipelines
    # run_other_pipeline(X_train, y_train, X_test, y_test, pfe_pipeline, "PFE", dataset_type, results)
    # run_other_pipeline(X_train, y_train, X_test, y_test, mi_pipeline, "MI", dataset_type, results)
    # run_other_pipeline(X_train, y_train, X_test, y_test, autoencoder_pipeline, "Autoencoder", dataset_type, results)

    # Once all runs are complete, hand off to visualization
    visualize_results(results, dataset_type)

    # Optionally print a summary
    print("\nAll runs complete. Results dictionary ready for analysis & plotting.")

if __name__ == "__main__":
    main()
