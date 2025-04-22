from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def train_model(clf, X_train, X_test, y_test, name="Model"):
    clf.fit(X_train.drop(columns='label'), X_train['label'])
    y_pred = clf.predict(X_test.drop(columns='label'))
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))

train_mlp = lambda X_train, X_test, y_test: train_model(MLPClassifier(max_iter=300), X_train, X_test, y_test, "MLP")