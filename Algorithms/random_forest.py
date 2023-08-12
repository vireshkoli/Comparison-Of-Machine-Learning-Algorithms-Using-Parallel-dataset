from collections import Counter

import numpy as np

from decision_tree import DecisionTree


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    import pandas as pd

    #data = pd.read_csv("Diabetes1.csv")
    data = pd.read_csv("Diabetes2.csv")
    #data = pd.read_csv("Diabetes3.csv")

    X = np.array(data.drop(['Outcome'],1))
    y = np.array(data['Outcome'])

    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix

    kfold = KFold(n_splits=20)

    score_rf = []

    for train_index, test_index in kfold.split(X):
        #print("Train : \n", train_index, "\nTest : \n", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf = RandomForest(n_trees=3, max_depth=5)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        score_rf.append(acc)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    print(score_rf)
    print("Accuracy:", (sum(score_rf) / len(score_rf)))

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1234
    # )

    # clf = RandomForest(n_trees=3, max_depth=5)

    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # acc = accuracy(y_test, y_pred)

    # print("Accuracy:", acc)
