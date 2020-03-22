import pandas as pd
from feature_extraction import extract_features, extract_text_features, create_column_with_text_without_mentions
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, recall_score, precision_score, f1_score

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# LOAD DATA
# read in individual datasets
trump_tweets = pd.read_json("Datasets/trump_tweets.json")
general_tweets = pd.read_json("Datasets/general_tweets.json")

# label the datasets
trump_tweets['label'] = 1
general_tweets['label'] = 0

# join the datasets
dataset = pd.DataFrame()
dataset = dataset.append(trump_tweets)
dataset = dataset.append(general_tweets)

def batch_predict(model, X_test):
    print("Predict model")
    X_test = X_test.tocsr()
    y_pred = []
    y_pred_proba = []
    rows = X_test.shape[0]
    prev_j = 0
    for j in range(1000, rows, 1000):
        X_test_batch = X_test[prev_j:j]
        y_pred.extend(model.predict(X_test_batch.toarray()))
        y_pred_proba.extend(model.predict_proba(X_test_batch.toarray()))
        prev_j = j

    if prev_j < rows:
        X_test_batch = X_test[prev_j:]
        y_pred.extend(model.predict(X_test_batch.toarray()))
        y_pred_proba.extend(model.predict_proba(X_test_batch.toarray()))

    return y_pred, np.array(y_pred_proba)


def run_learning_for_model(model_name, repeats):
    accuracies = []
    cms = []
    recalls = []
    precisions = []
    f1s = []
    aucs = []
    fprs =[]
    tprs =[]

    for i in range(repeats):
        print("Run:", i + 1)
        # split dataset into train and test datasets
        X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count', 'source', 'text', 'hashtags', 'symbols', 'user_mentions', 'media']], dataset['label'], test_size=0.20, random_state=0)
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)


        # REMOVE MENTIONS FROM TEXT
        X_train = create_column_with_text_without_mentions(X_train)
        X_test = create_column_with_text_without_mentions(X_test)


        # TEXT EXTRACTION
        X_train_tf, X_test_tf = extract_text_features(X_train, X_test, 'text', False)
        # X_train_tf, X_test_tf = extract_text_features(X_train, X_test, 'text_without_mentions')


        # EXTRACT FEATURES
        X_train = extract_features(X_train)
        X_test = extract_features(X_test)


        # DATA SCALING
        from sklearn.preprocessing import StandardScaler, Normalizer

        normalizer = Normalizer()
        normalizer.fit(X_train)
        X_train = normalizer.transform(X_train)
        X_test = normalizer.transform(X_test)


        # JOIN SCALED FEATURES AND TEXT MEASURES
        from scipy import sparse

        joined_train = sparse.hstack([X_train_tf, sparse.csr_matrix(X_train)])
        joined_test = sparse.hstack([X_test_tf, sparse.csr_matrix(X_test)])

        # RUN TRAINING
        from models import logistic_regression, naive_bayes, svm, calibrated_classifier, mlp

        print("Instantiate model")

        if model_name == 'logistic regression':
            model = logistic_regression(joined_train, y_train)
            y_pred, y_pred_proba = predict_model(model, joined_test)
            measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s,
                         accuracies, cms)
        elif model_name == 'naive bayes':
            model = naive_bayes(joined_train, y_train)
            y_pred, y_pred_proba = batch_predict(model, joined_test)
            measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s,
                         accuracies, cms)
        elif model_name == 'svm':
            model = svm(joined_train, y_train)
            y_pred, y_pred_proba = predict_model(model, joined_test)
            measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s,
                         accuracies, cms)
        elif model_name == 'calibrated classifier':
            model = calibrated_classifier(joined_train, y_train)
            y_pred, y_pred_proba = predict_model(model, joined_test)
            measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s,
                         accuracies, cms)
        elif model_name == 'mlp':
            model = mlp(joined_train, y_train)
            y_pred, y_pred_proba = predict_model(model, joined_test)
            measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s,
                         accuracies, cms)

    average_measurements(accuracies, recalls, precisions, f1s, aucs, fprs, tprs, model_name)


def predict_model(model, X_test):
    print("Predict model")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    return y_pred, y_pred_proba


def measurements(y_test, y_pred, y_pred_proba, fprs, tprs, aucs, recalls, precisions, f1s, accuracies, cms):
    # MEASUREMENTS
    print("Collecting Measurements")
    y_pred_proba = y_pred_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    fprs.append(fpr)
    tprs.append(tpr)


    auc_score = auc(fpr,tpr)
    aucs.append(auc_score)
    # print("AUC:", auc_score, '\n')

    recall = recall_score(y_test, y_pred)
    recalls.append(recall)
    # print("Recall:", recall, '\n')

    precision = precision_score(y_test,y_pred)
    precisions.append(precision)
    # print("Precision:", precision, '\n')

    f1 = f1_score(y_test,y_pred)
    f1s.append(f1)
    # print("f1:", f1, '\n')

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # print("Accuracy:", accuracy, '\n')

    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)
    # print(cm)

    # print(cms)
    # print(accuracies)


def average_measurements(accuracies, recalls, precisions, f1s, aucs, fprs, tprs, model_name):
    print("Average accuracy:", np.average(accuracies), '\n')
    print("Average recall:", np.average(recalls), '\n')
    print("Average precision:", np.average(precisions), '\n')
    print("Average f1:", np.average(f1s), '\n')

    avg_auc = np.average(aucs)
    print("Average AUCs:", avg_auc, '\n')

    avg_fpr = np.mean(fprs, axis=0)
    print("Average FPRs:", avg_fpr, '\n')

    avg_tpr = np.mean(tprs, axis=0)
    print("Average TPRs:", avg_tpr, '\n')

    plt.plot(avg_fpr, avg_tpr, label=f"{model_name} AUC = {round(avg_auc,5)}")


model_names = ['logistic regression', 'naive bayes', 'calibrated classifier']
repeats = 5
for model_name in model_names:
    print("Runs for:", model_name, '\n')
    run_learning_for_model(model_name, repeats)

plt.title(f"Average ROC Curve For {repeats} Runs")
plt.xlabel("Average False Positive Rate")
plt.ylabel("Average True Positive Rate")
plt.legend()
plt.show()


pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
