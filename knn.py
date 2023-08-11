from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    # Imports
    from matplotlib.colors import ListedColormap
    import pandas as pd
    from sklearn.model_selection import train_test_split


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    #data = pd.read_csv("Diabetes1.csv")
    data = pd.read_csv("Diabetes2.csv")
    #data = pd.read_csv("Diabetes3.csv")

    print(data)
    X = np.array(data.drop(['Outcome'],1))
    y = np.array(data['Outcome'])

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=1234
    # )


    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix

    kfold = KFold(n_splits=20)

    score_knn = []

    for train_index, test_index in kfold.split(X):
        #print("Train : \n", train_index, "\nTest : \n", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        k = 13 #Square root of Total Samples
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        a = accuracy(y_test, predictions)
        score_knn.append(a)
        cm = confusion_matrix(y_test, predictions)
        print(cm)
    print(score_knn)

    final_accuracy = sum(score_knn) / len(score_knn)
    print("KNN Accuracy = ", final_accuracy)

    

    # k = 13 #Square root of Total Samples
    # clf = KNN(k=k)
    # clf.fit(X_train, y_train)
    # predictions = clf.predict(X_test)
    # print("KNN classification accuracy", accuracy(y_test, predictions))