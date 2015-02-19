import numpy as np
import csv

# Number of samples in the dataset
NUM_TRAININGS = 4000

# NUmber of samples in the test dataset
NUM_TEST = 9868

# Filenames of training data
fin_names = {'raw':'data/kaggle_train_wc.csv', 'tf_idf':'data/kaggle_train_tf_idf.csv'}

# Filenames of test data
ftest_names = {'raw':'data/kaggle_test_wc.csv', 'tf_idf':'data/kaggle_test_tf_idf.csv'}

# Cross Validation 
K = 5

# Decision Tree parameters
criterions = ["gini", "entropy"]
splitters = ["best", "random"]
max_depths = [None] + range(1, 25)
min_samples_leafs = range(1, 100)
min_samples_splits = range(2, 500)
max_leaf_nodes = [None] + range(1, 100)

# SVM parameters
Cs = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
kernels = ['rbf', 'linear', 'poly', 'sigmoid']
degrees = range(1,5)
# gamma
# ceof0
# tol
# max_iter

# Return error rate
def get_error(G, Y):
    error = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
    return 1.0 * error / len(G)

# Return X_train and Y_train
def getXY(typeOfData):
    with open(fin_names[typeOfData], 'r') as fin:
        data = np.array(list(csv.reader(fin)))

    X = data[1:NUM_TRAININGS+1, 1:-1]
    Y = data[1:NUM_TRAININGS+1, -1]
    return X, Y

# Output submission file
def output_submission(typeOfData, clf, filename):
    # Retrieve train data
    X,Y = getXY(typeOfData)

    # Predict with the maximum model
    clf = clf.fit(X, Y)
    with open(ftest_names[typeOfData], 'r') as ftest:
        data = np.array(list(csv.reader(ftest)))

    # Write to submission file
    X_test = data[1:NUM_TEST+1, 1:]
    G = clf.predict(X_test)
    with open(filename, 'w') as predict:
        predict.write("Id,Prediction")
        for i in range(NUM_TEST):
            predict.write("\n{},{}", i, G[i])
