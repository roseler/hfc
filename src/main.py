from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from utils.unsw_loader import load_unsw_datasets
from utils.hfc import hfc_pipeline
from models.random_forest import train_rf
from models.knn import train_knn
from models.svm import train_svm
from models.mlp import train_mlp

def run_pipeline_with_clustering(clustering_method, name):
    print(f"\\n--- Running HFC with {name} Clustering ---")
    X_train, y_train, X_test, y_test = load_unsw_datasets("data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv")
    
    X_train_hfc = hfc_pipeline(X_train, y_train, clustering_method=clustering_method)
    X_test_hfc = hfc_pipeline(X_test, y_test, clustering_method=clustering_method)

    print("\\n[Random Forest]")
    train_rf(X_train_hfc, X_test_hfc, y_test)

    print("\\n[KNN]")
    train_knn(X_train_hfc, X_test_hfc, y_test)

    print("\\n[SVM]")
    train_svm(X_train_hfc, X_test_hfc, y_test)

    print("\\n[MLP]")
    train_mlp(X_train_hfc, X_test_hfc, y_test)

def main():
    clustering_methods = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=0.8, min_samples=5)
    }

    for name, method in clustering_methods.items():
        run_pipeline_with_clustering(method, name)

if __name__ == "__main__":
    main()