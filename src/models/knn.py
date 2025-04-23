from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def train_model(clf, X_train, y_train, X_test, y_test, name="Model"):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

train_knn = lambda X_train, X_test, y_train, y_test: train_model(
    KNeighborsClassifier(), X_train, y_train, X_test, y_test, "KNN"
)
