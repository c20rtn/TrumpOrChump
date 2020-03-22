import math


def batch_fit(model, X_train, y_train):
    X_train = X_train.tocsr()
    y_train = y_train.tolist()

    rows = X_train.shape[0]
    batches = int(math.ceil(rows / 1000))

    prev_i = 0
    for i in range(1000, rows, 1000):
        print(f"Batch {int(i / 1000)}/{batches}")
        X_batch = X_train[prev_i:i]
        y_batch = y_train[prev_i:i]
        model.partial_fit(X_batch.toarray(), y_batch, [0, 1])
        prev_i = i

    if prev_i < rows:
        print(f"Batch {batches}/{batches}")
        X_batch = X_train[prev_i:]
        y_batch = y_train[prev_i:]
        model.partial_fit(X_batch.toarray(), y_batch)

    return model


# LOGISTIC REGRESSION
def logistic_regression(X_train, y_train, folds):
    from sklearn.linear_model import LogisticRegressionCV

    lr_model = LogisticRegressionCV(cv=folds,
                                    random_state=0,
                                    max_iter=5000)
    print(f"Training model with {folds}-fold cross validation")
    return lr_model.fit(X_train, y_train)


# NAIVE BAYES
def naive_bayes(X_train, y_train, folds):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.calibration import CalibratedClassifierCV

    nb_model = MultinomialNB()
    nb_cc = CalibratedClassifierCV(nb_model, cv=folds)
    print(f"Training model with {folds}-fold cross validation")
    return nb_cc.fit(X_train, y_train)


# Calibrated Classifier
def svm(X_train, y_train, folds):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    svc_model = LinearSVC(random_state=0,
                          tol=1e-04,
                          max_iter=5000)
    svc_cc = CalibratedClassifierCV(svc_model, cv=folds)
    print(f"Training model with {folds}-fold cross validation")
    return svc_cc.fit(X_train, y_train)


# MLP
def mlp(X_train, y_train):
    from sklearn.neural_network import MLPClassifier

    mlp_model = MLPClassifier(random_state=0,
                              solver='adam',
                              hidden_layer_sizes=(1000, 250, 15, 8, 2),
                              tol=1e-6,
                              activation='relu',
                              max_iter=1000,
                              n_iter_no_change=20,
                              verbose=True,
                              learning_rate_init=0.01,
                              batch_size=500)
    print("Training model")
    return batch_fit(mlp_model, X_train, y_train)
