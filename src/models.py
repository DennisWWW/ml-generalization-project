from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


def run_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return {
        "model": "logistic_regression",
        "train_accuracy": accuracy_score(y_train, train_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "train_log_loss": log_loss(y_train, train_probs),
        "test_log_loss": log_loss(y_test, test_probs),
    }


def run_lda(X_train, X_test, y_train, y_test):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return {
        "model": "lda",
        "train_accuracy": accuracy_score(y_train, train_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "train_log_loss": log_loss(y_train, train_probs),
        "test_log_loss": log_loss(y_test, test_probs),
    }