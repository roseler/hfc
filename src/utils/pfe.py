import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def pfe_pipeline(X, degree=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    feature_names = poly.get_feature_names_out(X.columns)
    return pd.DataFrame(X_poly, columns=feature_names)