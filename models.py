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
def logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression(random_state=0, max_iter=5000)
    return lr_model.fit(X_train, y_train)


# NAIVE BAYES
def naive_bayes(X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB

    nb_model = MultinomialNB()
    return batch_fit(nb_model, X_train, y_train)


# SVM
def svm(X_train, y_train):
    from sklearn.svm import LinearSVC

    svc_model = LinearSVC(random_state=0, tol=1e-03, max_iter=5000)
    return svc_model.fit(X_train, y_train)


# Calibrated Classifier
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
                              solver='adam',
                              hidden_layer_sizes=(4000, 250, 15, 8, 2),
                              tol=1e-4,
                              activation='relu',
                              max_iter=200)
    return batch_fit(mlp_model, X_train, y_train)
