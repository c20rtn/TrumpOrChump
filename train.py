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

model_metrics = pd.DataFrame(
    columns=['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'FPR', 'TPR', 'AUC'])


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


def run_learning_for_model(model_name, metrics):
    accuracies = []
    cms = []
    recalls = []
    precisions = []
    f1s = []
    aucs = []
    fprs = []
    tprs = []

    # split dataset into train and test datasets
    # dataset = dataset.sample(frac=1)
    # print(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count',
                                                                 'source', 'text', 'hashtags', 'symbols',
                                                                 'user_mentions', 'media']], dataset['label'],
                                                        test_size=0.20, random_state=0, stratify=dataset['label'])
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
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # JOIN SCALED FEATURES AND TEXT MEASURES
    from scipy import sparse

    joined_train = sparse.hstack([X_train_tf, sparse.csr_matrix(X_train)])
    joined_test = sparse.hstack([X_test_tf, sparse.csr_matrix(X_test)])

    # RUN TRAINING
    from models import logistic_regression, naive_bayes, svm, mlp

    print("Instantiate model")

    if model_name == 'Logistic Regression':
        model = logistic_regression(joined_train, y_train, folds)
        y_pred, y_pred_proba = predict_model(model, joined_test)
        metrics = measurements(y_test, y_pred, y_pred_proba, model_name, metrics)
    elif model_name == 'Naive Bayes':
        model = naive_bayes(joined_train, y_train, folds)
        y_pred, y_pred_proba = batch_predict(model, joined_test)
        metrics = measurements(y_test, y_pred, y_pred_proba, model_name, metrics)
    elif model_name == 'SVM':
        model = svm(joined_train, y_train, folds)
        y_pred, y_pred_proba = predict_model(model, joined_test)
        metrics = measurements(y_test, y_pred, y_pred_proba, model_name, metrics)
    elif model_name == 'MLP':
        model = mlp(joined_train, y_train)
        y_pred, y_pred_proba = predict_model(model, joined_test)
        metrics = measurements(y_test, y_pred, y_pred_proba, model_name, metrics)

    print_metrics(metrics)


def predict_model(model, X_test):
    print("Predict model")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    return y_pred, y_pred_proba


def measurements(y_test, y_pred, y_pred_proba, model_name, metrics):
    # MEASUREMENTS
    print("Collecting Measurements")
    y_pred_proba = y_pred_proba[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    auc_score = auc(fpr, tpr)

    return metrics.append({'Model Name': model_name,
                           'Accuracy': accuracy,
                           'Precision': precision,
                           'Recall': recall,
                           'F1 Score': f1,
                           'Confusion Matrix': cm,
                           'FPR': fpr,
                           'TPR': tpr,
                           'AUC': auc_score},
                          ignore_index=True)


def print_metrics(metrics):
    print(metrics[['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'AUC']])

    for index, row in metrics.iterrows():
        plt.plot(row['FPR'], row['TPR'], label=f"{row['Model Name']} AUC = {round(row['AUC'], 5)}")


model_names = ['Logistic Regression', 'Naive Bayes', 'SVM']
folds = 5
# for model_name in model_names:
#     print("Runs for:", model_name, '\n')
#     run_learning_for_model(model_name, model_metrics)

run_learning_for_model('Naive Bayes', model_metrics)

plt.title(f"Best ROC Curve After {folds}-fold Cross Validation")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
