import numpy as np
import pandas as pd


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # data = pd.read_csv("Diabetes1.csv")
    # data = pd.read_csv("Diabetes2.csv")
    data = pd.read_csv("Diabetes3.csv")

    print(data)
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # features = ['BloodPressure', 'DiabetesPedigreeFunction']
    target = ['Outcome']

    

    X = data.loc[:, features].values
    print(X)
    y = np.array(data['Outcome'])
    y = np.where(y <= 0, -1, 1)

    kfold = KFold(n_splits=20)
    score_svm = []

    for train_index, test_index in kfold.split(X):
        #print("Train : \n", train_index, "\nTest : \n", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        k = 13 #Square root of Total Samples
        clf = SVM()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        a = accuracy(y_test, predictions)
        score_svm.append(a)
        cm = confusion_matrix(y_test, predictions)
        print(cm)
    print(score_svm)

    final_accuracy = sum(score_svm) / len(score_svm)
    print("SVM Accuracy = ", final_accuracy)

data = (4, 72, 90, 34, 200, 28.1, 0.287, 40)
scaler.fit(X)

#Converting to numpy array
data_array = np.asarray(data)

#Reshaping the array
data_reshape =  data_array.reshape(1,-1)

#Standardizing the data
data_standard = scaler.transform(data_reshape)

prediction = clf.predict(data_standard)


if(prediction[0] == 0):
    print('Not-diabetic')
else:
    print('Diabetic')


    