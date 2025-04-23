import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.unsw_loader import load_unsw_datasets
from utils.hfc import (
    hfc_pipeline, compute_feature_stats, suggest_hfc_parameters
)
from utils.pfe import pfe_pipeline
from utils.autoencoder import autoencoder_pipeline
from utils.mutual_info import mi_pipeline, mi_pipeline_transform
from models.random_forest import train_rf
from models.knn import train_knn
from models.mlp import train_mlp
from clustering.kmeans_runner import run_kmeans_hfc
from clustering.gmm_runner import run_gmm_hfc
from clustering.birch_runner import run_birch_hfc

def run_models(X_train, y_train, X_test, y_test, name=""):
    print(f"\n[{name} | Random Forest]")
    train_rf(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | KNN]")
    train_knn(X_train, X_test, y_train, y_test)

    print(f"\n[{name} | MLP]")
    train_mlp(X_train, X_test, y_train, y_test)

def main():
    train_path = "data/UNSW_NB15_training-set.csv"
    test_path = "data/UNSW_NB15_testing-set.csv"

    X_train, y_train, X_test, y_test = load_unsw_datasets(train_path, test_path)

    # === STEP 1: Auto-tune HFC parameters ===
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    feature_stats = compute_feature_stats(X_scaled)
    top_n, min_votes = suggest_hfc_parameters(feature_stats, cov_threshold=1.0)

    print(f"\nAuto-tuned Top-N: {top_n}, Min-Votes: {min_votes}")

    # === STEP 2: Run HFC with KMeans ===
    run_kmeans_hfc(X_train, y_train, X_test, y_test, min_votes)

    # === STEP 3: Run HFC with GMM ===
    run_gmm_hfc(X_train, y_train, X_test, y_test, min_votes)

    # === STEP 4: Run HFC with Birch ===
    run_birch_hfc(X_train, y_train, X_test, y_test, min_votes)

    # === STEP 5: Benchmarking Other Feature Engineering ===

    # PFE
    pfe_train = pfe_pipeline(X_train)
    pfe_train['label'] = y_train.values
    pfe_test = pfe_pipeline(X_test)
    pfe_test['label'] = y_test.values
    run_models(
        pfe_train.drop(columns=["label"]),
        y_train,
        pfe_test.drop(columns=["label"]),
        y_test,
        name="PFE"
    )

    # Mutual Information
    mi_train, mi_selector = mi_pipeline(X_train, y_train)
    mi_train['label'] = y_train.values
    mi_test = mi_pipeline_transform(X_test, mi_selector)
    mi_test['label'] = y_test.values
    run_models(
        mi_train.drop(columns=["label"]),
        y_train,
        mi_test.drop(columns=["label"]),
        y_test,
        name="MI"
    )

    # Autoencoder
    ae_train = autoencoder_pipeline(X_train)
    ae_train['label'] = y_train.values
    ae_test = autoencoder_pipeline(X_test)
    ae_test['label'] = y_test.values
    run_models(
        ae_train.drop(columns=["label"]),
        y_train,
        ae_test.drop(columns=["label"]),
        y_test,
        name="Autoencoder"
    )

if __name__ == "__main__":
    main()
