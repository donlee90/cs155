from sklearn import tree
from sklearn import cross_validation
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from CONFIG import *

fout_names = ['result/DT_raw.csv', 'result/DT_tf_idf.csv']
predict_names = ['predict/DT_raw.csv', 'predict/DT_tf_idf.csv']

# Make result directory
try:
    os.mkdir('result')
except OSError:
    pass
# Make result directory
try:
    os.mkdir('predict')
except OSError:
    pass
 
# Cross Validation
for i in range(2):
    typeOfData = 'raw' if i == 0 else 'tf_idf'
    print "---------------- {} data ----------------".format(typeOfData)

    # Retrieve train data
    X, Y = getXY(typeOfData)
    max_score = 0
    max_output = ""
    max_model = None
    fout = open(fout_names[i], 'w')

    for criterion in criterions:
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                # initialize the tree model
                clf = tree.DecisionTreeClassifier(
                        criterion=criterion, 
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf)

                # cross validate
                scores = cross_validation.cross_val_score(
                        clf, X, Y, cv=K, n_jobs=-1, scoring='accuracy')

                # Print result
                output = "criterion: {} / max_depth: {} / min_samples_leaf: {}\n".format(
                          criterion, max_depth, min_samples_leaf)
                output += "Scores = {}\nAccuracy = {:.2} (+/- {:.2})\n".format(
                          scores, scores.mean(), scores.std() * 2)
                print output
                fout.write(output)

                # Update max score
                if max_score < scores.mean():
                    max_score = scores.mean()
                    max_output = output
                    max_model = clf

    # Print max score
    print "**************** Maximum score ********************"
    print max_output
    fout.write("**************** Maximum score ********************")
    fout.write(max_output)

    # Output prediction
    output_submission(typeOfData, predict_names[i])

