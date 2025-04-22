import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

def mi_pipeline(X, y, k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)

    selected_features = selector.get_support(indices=True)
    selected_names = X.columns[selected_features]
    return pd.DataFrame(X_selected, columns=selected_names)