# 🎼 Harmonic Feature Chording (HFC)

**Harmonic Feature Chording (HFC)** is a contrast-driven feature engineering technique designed to extract **interpretable, high-impact feature interactions** from structured datasets. Inspired by the harmony of musical chords, HFC captures co-activating patterns of features that exhibit significant behavioral contrast within unsupervised clusters.

## Key Highlights
- **Cluster-Aware Modeling**: Uses KMeans, BIRCH, and GMM to uncover distinct behavioral segments.
- **Contrast-Based Scoring**: Identifies features that deviate meaningfully from global patterns.
- **Rule-Induced Feature Chords**: Generates compact binary features ("chords") that reflect co-activated, high-signal patterns.
- **Model-Agnostic Enhancement**: Compatible with classifiers like Random Forest, KNN, and MLP.
- **Lightweight & Interpretable**: Avoids black-box transformations for explainability and efficiency.

## Why HFC?
Traditional techniques like Polynomial Feature Expansion (PFE) indiscriminately create vast feature spaces. HFC selectively harmonizes features that matter, leading to:
- Reduced dimensionality
- Improved model interpretability
- Faster training & inference times

## 🔧 Cluster Configuration

To specify the number of clusters used by each clustering algorithm, edit the `main.py` script directly in the following lines:

```python
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


To further fine-tune chord generation, you can modify the following thresholds inside the `hfc_pipeline(...)` call within each `run_hfc_with_clustering(...)` block:

- `contrast_threshold`: Defines how distinct a feature’s behavior must be within a cluster to be included in a chord.
- `coverage_threshold`: Specifies the minimum proportion of instances a generated chord must cover to be retained.

These thresholds influence:

- **Sensitivity** → Lower `contrast_threshold` results in more features being included per chord.
- **Generality** → Lower `coverage_threshold` allows more specific, narrower chords to form.

This level of control enables you to adjust how broad or focused the extracted rules (chords) should be depending on the dataset or classification task.




## 📂 Structure
```bash
src/
│
├── dataset_loader/       # Loaders for IDS datasets (UNSW-NB15, CICIDS2017, etc.)
├── utils/                # HFC logic, monitoring, and preprocessing tools
├── models/               # Model wrappers for RF, KNN, and MLP
├── main.py               # Pipeline launcher
