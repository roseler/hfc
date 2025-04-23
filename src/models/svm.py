from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_model(clf, X_train, y_train, X_test, y_test, name="Model"):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

train_svm = lambda X_train, X_test, y_train, y_test: train_model(
    SVC(), X_train, y_train, X_test, y_test, "SVM"
)
