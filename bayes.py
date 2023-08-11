import pandas as pd

def loadCSV(filepath):
    print("-----------------LoadCsv functions---------------------")
    classList = []
    data = pd.read_csv(filepath)

    x = data.iloc[:,-1:].values
    print(x)
    for ele in x:
        classList.append(ele[0])
    print(classList, data)
    print("-----------------LoadCsv functions---------------------")
    return classList, data

def predictionClass(classList):
    print("--------------------predictionClass function-------------------")
    pred = {}
    for ele in set(classList):
        pred[ele] = 0
    for ele in classList: 
        pred[ele] += 1
    print(pred)
    print("--------------------predictionClass function-------------------")
    return pred

def getProbability(data, classList, pred):
    print("----------------------Get Probability function---------------------------")
    pred_data = pred.keys()
    print(pred_data)
    prob_data = {}
    labels = [col for col in data.columns]
    print(labels)
    del labels[-1]
    print(labels)
    for col in labels:
        prob_data[col] = {}
        for ele in set(list(data[col])):
            prob_data[col][ele] = {x : 0 for x in pred_data}
        for ele, prediction in zip(list(data[col]), classList):
            prob_data[col][ele][prediction] += 1
    print(prob_data)
    print("----------------------Get Probability function---------------------------")
    return prob_data

def getUnknownProbability(sample, probabilities, pred, classList):
    print("-------------------GetUnknownProbability function-------------------------")
    prob_list = {}
    for ele in pred.keys():
        prob_list[ele] = pred[ele] / len(classList)
        for col in sample.keys():
            prob_list[ele] *= probabilities[col][sample[col]][ele]  / pred[ele]
    print(prob_list)
    print("-------------------GetUnknownProbability function-------------------------")
    return prob_list

classList, data = loadCSV('./Diabetes1.csv')
print(classList)
pred = predictionClass(classList)
# prob_data = getProbability(data, classList, pred)
# unknown_sample = {'Outlook': 'Sunny', 'Temperature': 'hot', "Humidity": 'high', "Windy": False}
# print(f"The Unknown data sample is: {unknown_sample}")
# probs = getUnknownProbability(unknown_sample, prob_data, pred, classList)
# print(probs)

# key_list = list(probs.keys())
# val_list = list(probs.values())

# prob = max(val_list)
# res = key_list[val_list.index(prob)]
# for i in range(len(key_list)):
#     print(f"Class : {key_list[i]}, Probability: {val_list[i]}")
# print(f"Since the maximum probability is: {prob} of class: {res} therfore result is:\n{res} {prob}")
