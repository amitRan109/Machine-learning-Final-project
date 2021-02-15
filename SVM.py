from sklearn.svm import SVC

def run_svm( X_train, X_test, y_train, kernel_mode):
    # percent of success with:
    # 'linear' = 0.892
    # 'sigmoid' = 0.516
    # 'poly' = 0.88
    # 'rbf' = 0.864
    svclassifier = SVC(kernel=kernel_mode)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return y_pred
