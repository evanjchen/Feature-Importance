# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition   import PCA
from sklearn.ensemble        import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model    import LogisticRegression, RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold, cross_val_score
from   sklearn.pipeline           import Pipeline, FeatureUnion
from sklearn.base import clone
from   sklearn.metrics            import accuracy_score # We have not covered it yet in class. The basics - AUC is from 0 to 1 and higher is better.

import shap
from tqdm import tqdm
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
import rfpimp
import seaborn as sns
from scipy import stats
import shap


# Split to test, train, and
df = pd.read_csv('heart.csv')
df.columns
X = df.drop(columns='target')
y = df.target.values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

def get_spearman(X, shuffle=False):
    spearman = []
    for col in X.columns:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        values = X_train[col].values
        if shuffle==True:
            np.random.shuffle(y_train)
        spear_r = abs(stats.spearmanr(values, y_train)[0])
        spearman.append((col, spear_r))
    spearman.sort(key=lambda x: x[1], reverse=True)
    return spearman


def plot_spearman(features, spearman_values):
    plt.figure(figsize=(10,6))
    plt.barh(features,spearman_values)
    plt.title("Spearman's R for Features", fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Spearman's R", fontsize=15)
    plt.show()

def plot_importances(bench_avg, col_scores):
    """Input: Tuple of features and their metric scores;  benchmark model score"""
    # Subtract the accuracy with column removed from the benchmark
    diffs = [(x[0], bench_avg - x[1]) for x in col_scores]

    # Sort and plot
    diffs.sort(key=lambda x:x[1])
    features = [x[0] for x in diffs]
    importances = [x[1] for x in diffs]

    plt.figure(figsize=(10,6))
    plt.barh(features,importances)
    plt.title("Feature Importances", fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Accuracy Score difference", fontsize=15)
    plt.show()


def drop_col(model=None):
    """ Output: Average prediction accuracy of benchmark and dropped columns"""
    dropped_scores = []
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    bench_score = accuracy_score(y_val, y_pred)

    # Clone the model so the classifier's parameters are the same
    for col in X_train.columns:
        cloned = clone(model)
        X_train_2 = X_train.drop(columns=col)
        X_val_2 = X_val.drop(columns=col)
        cloned.fit(X_train_2, y_train)
        score = accuracy_score(y_val, cloned.predict(X_val_2))
        dropped_scores.append(score)
    return bench_score, np.array(dropped_scores)

def permute_col(X, y, model=None):

    permuted_scores = []
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    bench_score = accuracy_score(y_val, y_pred)

    # Permute the values in a column
    for col in X_train.columns:
        cloned = clone(model)
        # Randomly permute the training values and reassign
        X_val_2 = X_val.copy()
        col_vals = X_val[col].values
        np.random.shuffle(col_vals)
        X_val_2[col] = col_vals
        score = accuracy_score(y_val, model.predict(X_val_2))
        permuted_scores.append(score)
    return bench_score, np.array(permuted_scores)


    # Get the top features
def get_best_features(importances, k, reverse=True):
    """Input: Sort list of columns and their feature importances.
    Output: top k features"""
    importances = sorted(importances, key=lambda x: x[1], reverse=reverse)
    top_features = [x[0] for x in importances[:k]]
    return top_features


# retrain model
def selected_features_score(X, y, new_features, model=None, n=10):
    """Returns avg scores of model trained on new features"""
    X = X[new_features]
    scores = []
    for i in range(n):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)
    return np.mean(scores)

def compare_strat(k, PCA_features, spearman, dropped_avg, permute_avgs):
    """Input: k = num_features; lists of tuples of (col, score)"""
    """Output: Accuracy score for 1 to k features selected for each feature importance method"""
    all_scores = []
    selected_features = []
    for i in range(1,k+1):
        feat_spear = get_best_features(spearman, i)
        feat_drop =  get_best_features(dropped_avg, i, reverse=False)
        feat_perm = get_best_features(permute_avgs, i, reverse=False)
        feat_PCA = PCA_features[:i]
        selected_features.append((feat_spear, feat_drop, feat_perm, feat_PCA))
        all_featimp = [feat_spear, feat_drop, feat_perm, feat_PCA]
        accuracy = []
        for features in all_featimp:
            score = selected_features_score(X, y, features, model=RandomForestClassifier(), n=10)
            accuracy.append(score)
        all_scores.append(accuracy)
    return all_scores, selected_features


def plot_methods(k, all_scores, labels):
    """Plot the accuracy error for when k features are chose from (0,k)"""
    x = range(1, k+1)
    plt.figure(figsize=(10,10))
    for i in range(len(all_scores[0])):
        plt.plot(x, [1 - point[i] for point in all_scores], label=labels[i], linewidth=4, marker = 'o')
    plt.legend()
    plt.title('Accuracy Error for Different methods of Feature Importance Using Random Forest', fontsize=16, fontweight='bold')
    plt.xlabel("# Features Selected", fontsize=16, fontweight='bold')
    plt.ylabel("Prediction Accuracy Error", fontsize=16, fontweight='bold')
    plt.show()


def remove_features(bench_avg):
    """Keep on removing features until the score is less than the benchmark"""
    print("BENCHMARK:", bench_avg)
    current_features = X.columns
    spearman = get_spearman(X)
    scores = []
    for i in range(len(X.columns)):
        spear_cols = [x[0] for x in spearman]
        new_features = spear_cols[:-1]
        score = selected_features_score(X, y, new_features, model=RandomForestClassifier(), n=10)
        print("SCORE:", score)
        scores.append(score)
        if score >=bench_avg:
            spearman = get_spearman(X[new_features])
            current_features = new_features
        else:
            break
    return current_features, scores



def plot_feature_removal(scores, bench_avg):
    """Plot the results of the iterative feature removal"""
    x = range(1, len(scores)+1)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x, scores, color='#add8e6')
    plt.title("Accuracy Score after Removing Features", fontweight='bold', fontsize = 16)
    plt.ylabel("Accuracy Score", fontweight='bold', fontsize = 16)
    plt.xlabel("# Of Features Removed", fontweight='bold', fontsize = 16)
    ax.set_xticks(x)
    plt.axhline(y=bench_avg, color='brown')
    plt.text(0.8, 0.83, f"BENCHMARK: {round(bench_avg, 3)}", color='brown', fontsize=12)
    for i, score in enumerate(scores):
        color = 'green'
        if i ==len(scores)-1:
            color = 'red'
        plt.text(i+1-0.1, 0.75, f"{round(score, 3)}", color=color, fontsize=12)
    plt.show()


def get_feature_variance(X):
    """Gets mean and standard deviation for feature importance over iterations"""

    dic = {x:[] for x in X.columns}
    for x in range(100):
        spearman = get_spearman(X)  # call function to get new spearman's coeficient
        for col, corr in spearman:
            dic[col].append(corr)
    normalized = []
    for k, v in dic.items():
        mean = np.mean(v)
        variance = np.std(v)
        normalized.append((k, mean, variance))
    normalized.sort(key=lambda x:x[1], reverse=True)
    return normalized


def plot_feature_variance(normalized):
    """Plot feature importances with variances"""
    cols, means, variances = [], [], []
    for x in reversed(normalized):
        cols.append(x[0])
        means.append(x[1])
        variances.append(x[2]*2)
    fig, ax = plt.subplots(figsize=(12,8))
    plt.barh(cols, means, xerr=variances, capsize=10)
    plt.title("Spearman's R with 2 Standard Deviations", fontweight='bold', fontsize=16)
    plt.xlabel("Spearman's R", fontweight='bold', fontsize=16)
    ax.set_yticklabels(cols, fontsize=16)

    plt.xlim(0,1)
    plt.show()

def permute_y(X, n):
    """Input: n = iterations
    Output: Returns dictionary of % randomly permuting y-columns greater than true importance"""
    baseline_spear = get_spearman(X)
    baseline_spear = {x[0]:x[1] for x in baseline_spear}
    dic_pval = {col: 0 for col in X.columns}

    for x in range(n):
        spearman = get_spearman(X, shuffle=True)
        for col, corr in spearman:
            if corr > baseline_spear[col]:
                dic_pval[col] +=1
    for k, v in dic_pval.items():
        dic_pval[k] = v/n
    return dic_pval

def plot_pval(dic_pval):

    x = list(dic_pval.keys())
    y = list(dic_pval.values())

    fig, ax = plt.subplots(figsize=(12,8))
    plt.barh(x, y)
    plt.title("Fraction of Randomly Permuting y-columns is Greater than True Feature Importance", fontweight='bold', fontsize=16)
    plt.xlabel("Fraction greater than true importance", fontweight='bold', fontsize=16)
    ax.set_yticklabels(x, fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim(0,1)
    plt.show()
