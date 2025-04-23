import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_loader.unsw_loader import load_unsw_datasets
from dataset_loader.ids2017_loader import load_ids2017_dataset
from dataset_loader.ids2018_loader import load_ids2018_dataset

from utils.hfc import hfc_pipeline, compute_feature_stats, suggest_hfc_parameters
from utils.hfc_pipeline_cic2017 import hfc_pipeline_fit_cic2017, hfc_pipeline_transform_cic2017
from utils.hfc_pipeline_cse2018 import hfc_pipeline_fit_cse2018, hfc_pipeline_transform_cse2018

from utils.pfe import pfe_pipeline
from utils.autoencoder import autoencoder_pipeline
from utils.mutual_info import mi_pipeline, mi_pipeline_transform

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


def run_hfc_custom(X_train, y_train, X_test, y_test, dataset_type):
    print(f"\n--- HFC Feature Engineering for {dataset_type} ---")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    feature_stats = compute_feature_stats(X_scaled)
    _, min_votes = suggest_hfc_parameters(feature_stats, cov_threshold=1.0)

    if dataset_type == "UNSW":
        from clustering.kmeans_runner import run_kmeans_hfc
        from clustering.gmm_runner import run_gmm_hfc
        from clustering.birch_runner import run_birch_hfc

        run_kmeans_hfc(X_train, y_train, X_test, y_test, min_votes)
        run_gmm_hfc(X_train, y_train, X_test, y_test, min_votes)
        run_birch_hfc(X_train, y_train, X_test, y_test, min_votes)

    elif dataset_type == "CICIDS2017":
        from utils.hfc_pipeline_cic2017 import hfc_pipeline_fit_cic2017, hfc_pipeline_transform_cic2017

        motifs_train, chord_map, rule_map, coverage_map, scaler = hfc_pipeline_fit_cic2017(
            X_train, y_train, contrast_threshold=0.5, cov_threshold=1.0, min_votes=min_votes)

        motifs_test = hfc_pipeline_transform_cic2017(X_test, rule_map, scaler)

        # Align test features to train
        motifs_test = motifs_test.reindex(columns=motifs_train.drop(columns=["label"]).columns, fill_value=0)

        run_models(motifs_train.drop(columns=["label"]), y_train, motifs_test, y_test, name="HFC-CIC2017")

    elif dataset_type == "CSE-CIC-IDS2018":
        from utils.hfc_pipeline_cse2018 import hfc_pipeline_fit_cse2018, hfc_pipeline_transform_cse2018

        motifs_train, chord_map, rule_map, coverage_map, scaler = hfc_pipeline_fit_cse2018(
            X_train, y_train, contrast_threshold=0.5, cov_threshold=1.0, min_votes=min_votes)

        motifs_test = hfc_pipeline_transform_cse2018(X_test, rule_map, scaler)

        # Align test features to train
        motifs_test = motifs_test.reindex(columns=motifs_train.drop(columns=["label"]).columns, fill_value=0)

        run_models(motifs_train.drop(columns=["label"]), y_train, motifs_test, y_test, name="HFC-CSE2018")


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
        X_train, X_test, y_train, y_test = load_ids2018_dataset("data/CSE-CIC-IDS2018 (02-14-2018)_2.csv")
        dataset_type = "CSE-CIC-IDS2018"
    else:
        print("Invalid choice.")
        return

    # === Run dataset-specific HFC ===
    run_hfc_custom(X_train, y_train, X_test, y_test, dataset_type)

    # === PFE
    pfe_train = pfe_pipeline(X_train)
    pfe_train['label'] = y_train.values
    pfe_test = pfe_pipeline(X_test)
    pfe_test['label'] = y_test.values
    run_models(pfe_train.drop(columns=["label"]), y_train, pfe_test.drop(columns=["label"]), y_test, name="PFE")

    # === Mutual Information
    mi_train, mi_selector = mi_pipeline(X_train, y_train)
    mi_train['label'] = y_train.values
    mi_test = mi_pipeline_transform(X_test, mi_selector)
    mi_test['label'] = y_test.values
    run_models(mi_train.drop(columns=["label"]), y_train, mi_test.drop(columns=["label"]), y_test, name="MI")

    # === Autoencoder
    ae_train = autoencoder_pipeline(X_train)
    ae_train['label'] = y_train.values
    ae_test = autoencoder_pipeline(X_test)
    ae_test['label'] = y_test.values
    run_models(ae_train.drop(columns=["label"]), y_train, ae_test.drop(columns=["label"]), y_test, name="Autoencoder")


if __name__ == "__main__":
    main()
