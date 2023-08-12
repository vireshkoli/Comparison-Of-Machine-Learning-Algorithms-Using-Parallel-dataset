import pandas as pd
from pprint import pprint

def loadCSV(filepath):
    classList = []
    data = pd.read_csv(filepath)

    x = data.iloc[:,-1:].values
    for ele in x:
        classList.append(ele[0])
    return classList, data

def predictionClass(classList):
    pred = {}
    for ele in set(classList):
        pred[ele] = 0
    for ele in classList: 
        pred[ele] += 1
    return pred

def getProbability(data, classList, pred):
    pred_data = pred.keys()
    prob_data = {}
    labels = [col for col in data.columns]
    del labels[-1]
    for col in labels:
        prob_data[col] = {}
        for ele in set(list(data[col])):
            prob_data[col][ele] = {x : 0 for x in pred_data}
        for ele, prediction in zip(list(data[col]), classList):
            prob_data[col][ele][prediction] += 1
    return prob_data, labels


def getUnknownProbability(sample, probabilities, pred, classList):
    prob_list = {}
    for ele in pred.keys():
        prob_list[ele] = pred[ele] / len(classList)
        for col in sample.keys():
            prob_list[ele] *= probabilities[col][sample[col]][ele]  / pred[ele]
    return prob_list

def inTrain(curr_data, train_col):
    if curr_data in train_col:
        return curr_data
    si = 9999999
    for ele in train_col:
        if abs(curr_data - ele) < si:
            si = abs(curr_data - ele)
            curr_data = ele
    return curr_data

def getResultantClass(probs):
    outcome_class = list(probs.keys())
    outcome_class_value = list(probs.values())

    res_class_value = max(outcome_class_value)
    return outcome_class[outcome_class_value.index(res_class_value)]

def getAccuracy(predicted_classes, actual_classes):
    correct_prediction = 0
    for prediction, actual in zip(predicted_classes, actual_classes):
        if prediction == actual:
            correct_prediction += 1
    accur = (correct_prediction / len(actual_classes)) * 100
    return accur

classList1, data1 = loadCSV('./Diabetes1.csv')
classList2, data2 = loadCSV('./Diabetes2.csv')
classList3, data3 = loadCSV('./Diabetes3.csv')

train_data_outcome1, test_data_outcome1 = classList1[:200], classList1[201:]
train_data1, test_data1 = data1[:200], data1[201:]

train_data_outcome2, test_data_outcome2 = classList2[:200], classList2[201:]
train_data2, test_data2 = data2[:200], data2[201:]

train_data_outcome3, test_data_outcome3 = classList3[:200], classList3[201:]
train_data3, test_data3 = data3[:200], data3[201:]


train_outcome_class1 = predictionClass(train_data_outcome1)
train_outcome_class2 = predictionClass(train_data_outcome2)
train_outcome_class3 = predictionClass(train_data_outcome3)


train_data_analysis1, labels1 = getProbability(train_data1, train_data_outcome1, train_outcome_class1)
train_data_analysis2, labels2 = getProbability(train_data2, train_data_outcome2, train_outcome_class2)
train_data_analysis3, labels3 = getProbability(train_data3, train_data_outcome3, train_outcome_class3)

test_classes = []
for ind in test_data1.index:
    unknown_sample = {}
    for label in labels1:
        unknown_sample[label]  = inTrain(test_data1[label][ind], list(train_data1[label].values))
    test_data_outcome = getUnknownProbability(unknown_sample, train_data_analysis1, train_outcome_class1, train_data_outcome1)
    test_classes.append(getResultantClass(test_data_outcome))

accuracy = getAccuracy(test_classes, test_data_outcome1)
print(f"Accuracy for first dataset is: {accuracy}%")

test_classes = []
for ind in test_data2.index:
    unknown_sample = {}
    for label in labels2:
        unknown_sample[label]  = inTrain(test_data2[label][ind], list(train_data2[label].values))
    test_data_outcome = getUnknownProbability(unknown_sample, train_data_analysis2, train_outcome_class2, train_data_outcome2)
    test_classes.append(getResultantClass(test_data_outcome))

accuracy = getAccuracy(test_classes, test_data_outcome2)
print(f"Accuracy for Second dataset is: {accuracy}%")

test_classes = []
for ind in test_data3.index:
    unknown_sample = {}
    for label in labels3:
        unknown_sample[label]  = inTrain(test_data3[label][ind], list(train_data3[label].values))
    test_data_outcome = getUnknownProbability(unknown_sample, train_data_analysis3, train_outcome_class3, train_data_outcome3)
    test_classes.append(getResultantClass(test_data_outcome))

accuracy = getAccuracy(test_classes, test_data_outcome3)
print(f"Accuracy for third dataset is: {accuracy}%")