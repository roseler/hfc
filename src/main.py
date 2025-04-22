from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture

from utils.unsw_loader import load_unsw_datasets
from utils.hfc import hfc_pipeline
from utils.pfe import pfe_pipeline
from utils.autoencoder import autoencoder_pipeline
from utils.mutual_info import mi_pipeline, mi_pipeline_transform

from models.random_forest import train_rf
from models.knn import train_knn
from models.svm import train_svm
from models.mlp import train_mlp

def run_models(X_train, y_train, X_test, y_test, name=""):
    print(f"\n[{name} | Random Forest]")
    train_rf(X_train, X_test, y_test)

    print(f"\n[{name} | KNN]")
    train_knn(X_train, X_test, y_test)

    print(f"\n[{name} | SVM]")
    train_svm(X_train, X_test, y_test)

    print(f"\n[{name} | MLP]")
    train_mlp(X_train, X_test, y_test)

def main():
    train_path = "data/UNSW_NB15_training-set.csv"
    test_path = "data/UNSW_NB15_testing-set.csv"
    
    X_train, y_train, X_test, y_test = load_unsw_datasets(train_path, test_path)

    # HFC - with clustering variations
    clustering_methods = {
        "HFC-KMeans": KMeans(n_clusters=2, random_state=42),
        "HFC-Birch": Birch(n_clusters=2),
        "HFC-GMM": GaussianMixture(n_components=2, random_state=42)
    }

    for name, clusterer in clustering_methods.items():
        hfc_train = hfc_pipeline(X_train, y_train, clustering_method=clusterer)
        hfc_test = hfc_pipeline(X_test, y_test, clustering_method=clusterer)
        run_models(hfc_train, y_train, hfc_test, y_test, name=name)

    # Polynomial Feature Expansion
    pfe_train = pfe_pipeline(X_train)
    pfe_train['label'] = y_train.values
    pfe_test = pfe_pipeline(X_test)
    pfe_test['label'] = y_test.values
    run_models(pfe_train, y_train, pfe_test, y_test, name="PFE")

    # Mutual Information Selection
    mi_train, mi_selector = mi_pipeline(X_train, y_train)
    mi_train['label'] = y_train.values
    mi_test = mi_pipeline_transform(X_test, mi_selector)
    mi_test['label'] = y_test.values
    run_models(mi_train, y_train, mi_test, y_test, name="MI")

    # Autoencoder Features
    ae_train = autoencoder_pipeline(X_train)
    ae_train['label'] = y_train.values
    ae_test = autoencoder_pipeline(X_test)
    ae_test['label'] = y_test.values
    run_models(ae_train, y_train, ae_test, y_test, name="Autoencoder")

if __name__ == "__main__":
    main()
