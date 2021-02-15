from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def run_KNN(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    knn_model = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    return y_pred

