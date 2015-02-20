import numpy as np
import csv
from CONFIG import *

# Return error rate
def get_error(G, Y):
    error = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
        if G[i] == 1 and Y[i] == 1:
            true_pos += 1
        elif G[i] == 0 and Y[i] == 0:
            true_neg += 1
        elif G[i] == 0 and Y[i] == 1:
            false_pos += 1
        else:
            false_neg += 1

    error = 1.0 * error / len(G)
    precision = 1.0 * true_pos / (true_pos + false_pos)
    recall = 1.0 * true_pos / (true_pos + false_neg)
    return  error, precision, recall

# Return X_train and Y_train
def getTrainData(typeOfData):
    with open(fin_names[typeOfData], 'r') as fin:
        data = np.array(list(csv.reader(fin)))

    X = data[1:NUM_TRAININGS+1, 1:-1].astype(float)
    Y = data[1:NUM_TRAININGS+1, -1].astype(float)
    return X, Y


# Return X_test
def getTestData(typeOfData):
    with open(ftest_names[typeOfData], 'r') as ftest:
        data = np.array(list(csv.reader(ftest)))
    return data[1:NUM_TEST+1, 1:]


# Output submission file
def output_submission(typeOfData, clf, filename):
    X_test = getTestData(typeOfData)
    G = clf.predict(X_test)
    with open(filename, 'w') as predict_f:
        predict_f.write("Id,Prediction")
        for i in range(NUM_TEST):
            predict_f.write("\n{},{}".format(i, G[i]))
