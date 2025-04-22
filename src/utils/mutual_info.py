import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

def mi_pipeline(X_train, y_train, k=10):
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = selector.get_feature_names_out(X_train.columns)
    return pd.DataFrame(X_train_selected, columns=selected_features), selector

def mi_pipeline_transform(X, selector):
    X_selected = selector.transform(X)
    selected_features = selector.get_feature_names_out()
    return pd.DataFrame(X_selected, columns=selected_features)