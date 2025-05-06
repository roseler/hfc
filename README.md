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

## 📂 Structure
```bash
src/
│
├── dataset_loader/       # Loaders for IDS datasets (UNSW-NB15, CICIDS2017, etc.)
├── utils/                # HFC logic, monitoring, and preprocessing tools
├── models/               # Model wrappers for RF, KNN, and MLP
├── main.py               # Pipeline launcher
