from sklearn.linear_model import LogisticRegression

def run_log( X_train, X_test, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred
