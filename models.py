import math
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


def batch_fit(model, X_train, y_train, batch_size):
    rows = X_train.shape[0]
    batches = int(math.ceil(rows / batch_size))

    prev_i = 0
    for i in range(batch_size, rows, batch_size):
        print(f"Batch {int(i / batch_size)}/{batches}")
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


def logistic_regression(X_train_val, y_train_val, folds):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=0.2,
                                                      random_state=0,
                                                      stratify=y_train_val)

    if folds == 1:
        lr_model = LogisticRegression(random_state=0, max_iter=5000)
        print("Training model")
        lr_model = lr_model.fit(X_train, y_train)
        lr_cc = CalibratedClassifierCV(lr_model, cv='prefit')
        return lr_cc.fit(X_val, y_val)
    else:
        lr_model = LogisticRegressionCV(cv=folds,
                                        random_state=0,
                                        max_iter=5000)
        print(f"Training model with {folds}-fold cross validation")
        return lr_model.fit(X_train_val, y_train_val)


def naive_bayes(X_train_val, y_train_val, folds):
    from sklearn.naive_bayes import MultinomialNB

    nb_model = MultinomialNB()

    if folds == 1:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                          y_train_val,
                                                          test_size=0.2,
                                                          random_state=0,
                                                          stratify=y_train_val)

        print("Training model")
        nb_model = batch_fit(nb_model, X_train.tocsr(), y_train.tolist(), 1000)
        nb_cc = CalibratedClassifierCV(nb_model, cv='prefit')
        return nb_cc.fit(X_val, y_val)
    else:
        nb_cc = CalibratedClassifierCV(nb_model, cv=folds)
        print(f"Training model with {folds}-fold cross validation")
        return nb_cc.fit(X_train_val, y_train_val)


def svm(X_train_val, y_train_val, folds):
    from sklearn.svm import LinearSVC

    svc_model = LinearSVC(random_state=0,
                          tol=1e-04,
                          max_iter=5000)

    if folds == 1:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                          y_train_val,
                                                          test_size=0.2,
                                                          random_state=0,
                                                          stratify=y_train_val)

        print("Training model")
        svc_model.fit(X_train, y_train)
        svc_cc = CalibratedClassifierCV(svc_model, cv='prefit')
        return svc_cc.fit(X_val, y_val)
    else:
        svc_cc = CalibratedClassifierCV(svc_model, cv=folds)
        print(f"Training model with {folds}-fold cross validation")
        return svc_cc.fit(X_train_val, y_train_val)


def mlp(X_train_val, y_train_val, folds):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import KFold

    mlp_model = MLPClassifier(random_state=0,
                              solver='adam',
                              hidden_layer_sizes=(1000, 250, 15, 8, 2),
                              tol=1e-6,
                              activation='relu',
                              max_iter=1000,
                              n_iter_no_change=20,
                              learning_rate_init=0.01,
                              batch_size=500,
                              verbose=True,
                              warm_start=True)

    if folds == 1:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                          y_train_val,
                                                          test_size=0.2,
                                                          random_state=0,
                                                          stratify=y_train_val)
        X_train = X_train.tocsr()
        y_train = y_train.to_numpy()

        print("Training model")
        mlp_model = batch_fit(mlp_model, X_train, y_train, 1000)
        mlp_cc = CalibratedClassifierCV(mlp_model, cv='prefit')
        return mlp_cc.fit(X_val, y_val)
    else:
        X_train_val = X_train_val.tocsr()
        y_train_val = y_train_val.to_numpy()

        kf = KFold(n_splits=folds)
        print(f"Training model with {folds}-fold cross validation")
        for train_indices, test_indices in kf.split(X_train_val):
            mlp_model = batch_fit(mlp_model, X_train_val[train_indices], y_train_val[train_indices], 1000)

        return mlp_model
