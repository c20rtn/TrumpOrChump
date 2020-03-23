import pandas as pd
from feature_extraction import extract_features, extract_text_features, create_text_without_mentions
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, recall_score, precision_score, f1_score
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
from models import logistic_regression, naive_bayes, svm, mlp

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# read in individual datasets
trump_tweets = pd.read_json("Datasets/trump_tweets.json")
chump_tweets = pd.read_json("Datasets/general_tweets.json")

# label the datasets
trump_tweets['label'] = 1
chump_tweets['label'] = 0

# join the datasets
dataset = pd.DataFrame()
dataset = dataset.append(trump_tweets)
dataset = dataset.append(chump_tweets)

# create a dataframe for storing metrics
model_metrics = pd.DataFrame(
    columns=['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'FPR', 'TPR', 'AUC'])


def run_learning_for_model(model_name, metrics):
    print("Running", model_name)

    print("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(dataset[['favorite_count', 'is_quote_status', 'retweet_count',
                                                                 'source', 'text', 'hashtags', 'symbols',
                                                                 'user_mentions', 'media']], dataset['label'],
                                                        test_size=0.20, random_state=0, stratify=dataset['label'])
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    # create a column containing the text without user mentions
    X_train = create_text_without_mentions(X_train)
    X_test = create_text_without_mentions(X_test)

    print("Extracting text features")
    X_train_tf, X_test_tf = extract_text_features(X_train, X_test, False)

    print("Extracting other features")
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    print("Scaling data")
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("Combining features")
    joined_train = sparse.hstack([X_train_tf, sparse.csr_matrix(X_train)])
    joined_test = sparse.hstack([X_test_tf, sparse.csr_matrix(X_test)])

    print("Instantiating model")
    if model_name == 'Logistic Regression':
        model = logistic_regression(joined_train, y_train, folds)
    elif model_name == 'Naive Bayes':
        model = naive_bayes(joined_train, y_train, folds)
    elif model_name == 'SVM':
        model = svm(joined_train, y_train, folds)
    elif model_name == 'MLP':
        model = mlp(joined_train, y_train, folds)
    else:
        model = logistic_regression(joined_train, y_train, folds)

    dump(model, f"{model_name}-{folds}-folds.joblib")
    y_pred, y_pred_proba = predict_model(model, joined_test)
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, model_name, metrics)

    return metrics


def predict_model(model, X_test):
    print("Evaluating model performance", '\n')
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    return y_pred, y_pred_proba


def calculate_metrics(y_test, y_pred, y_pred_proba, model_name, metrics):
    y_pred_proba = y_pred_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

    return metrics.append({'Model Name': model_name,
                           'Accuracy': accuracy_score(y_test, y_pred),
                           'Precision': precision_score(y_test, y_pred),
                           'Recall': recall_score(y_test, y_pred),
                           'F1 Score': f1_score(y_test, y_pred),
                           'Confusion Matrix': confusion_matrix(y_test, y_pred),
                           'FPR': fpr,
                           'TPR': tpr,
                           'AUC': auc(fpr, tpr)},
                          ignore_index=True)


def print_metrics(metrics):
    print("Measured metrics")
    print(metrics[['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix', 'AUC']])

    for index, row in metrics.iterrows():
        plt.plot(row['FPR'], row['TPR'], label=f"{row['Model Name']} AUC = {round(row['AUC'], 5)}")

    if folds == 1:
        plt.title("ROC Curve")
    else:
        plt.title(f"ROC Curve After {folds}-fold Cross Validation")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.savefig(f"ROC-curve-{folds}-folds.png", dpi=100)
    plt.show()


model_names = ['Logistic Regression', 'Naive Bayes', 'SVM', 'MLP']
folds = 1
for model_name in model_names:
    model_metrics = run_learning_for_model(model_name, model_metrics)

print_metrics(model_metrics)

pd.reset_option('display.max_colwidth')
pd.reset_option('display.max_columns')
