# LOGISTIC REGRESSION
def logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(random_state=0, max_iter=5000)
    return lr_model.fit(X_train, y_train)


# NAIVE BAYES
def naive_bayes(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB

    nb_model = GaussianNB()
    return nb_model.fit(X_train, y_train)


# SVM
def svm(X_train, y_train):
    from sklearn.svm import LinearSVC

    svc_model = LinearSVC(random_state=0, tol=1e-03, max_iter=5000)
    return svc_model.fit(X_train, y_train)


def calibrated_classifier(X_train, y_train):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    svc_model = LinearSVC(random_state=0, tol=1e-03, max_iter=5000)
    cc = CalibratedClassifierCV(svc_model)
    return cc.fit(X_train, y_train)


# MLP
def mlp(X_train, y_train):
    from sklearn.neural_network import MLPClassifier

    mlp_model = MLPClassifier(random_state=0,
                            solver='lbfgs',
                            hidden_layer_sizes=(5, 2),
                            tol = 1e-4,
                            activation = 'relu',
                            max_iter = 200
                            )
    return mlp_model.fit(X_train, y_train)
