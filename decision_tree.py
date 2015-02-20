from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import os

from CONFIG import *
from common import *

fout_names = {'raw':'result/DT_raw.csv', 'tf_idf':'result/DT_tf_idf.csv'}
predict_names = {'raw':'predict/DT_raw', 'tf_idf':'predict/DT_tf_idf'}

# Make result directory
try:
    os.mkdir('result')
except OSError:
    pass

# Make predict directory
try:
    os.mkdir('predict')
except OSError:
    pass
 
# Finding the best parameter set
for typeOfData in types:
    print "---------------- ",
    print typeOfData.upper(),
    print " ----------------"

    # retrieve train data
    X, Y = getTrainData(typeOfData)
    X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

    # extensive grid search for best estimator
    for score in scores:
        # grid search
        print ("Tuning hyper-parameters for %s...\n" % score)
        clf = GridSearchCV(DecisionTreeClassifier(), 
                param_grid_DT, cv=K, scoring=score)

        # train the model
        print "Training for the entire data...\n"
        clf = clf.fit(X, Y)
        
        # Print final statistics
        print "****** Statistics ******\n"
        output = "Best score: "
        output += str(clf.best_score_)
        output += "\nBest parameters set found on development set: \n"
        output += str(clf.best_estimator_)
        output += "\n\nGrid scores on development set: \n"
        for params, mean_score, grid_scores in clf.grid_scores_:
            output += ("%0.3f (+/- %0.03f) for %r\n" % 
            (mean_score, grid_scores.std() / 2, params))
        print output
        print "******* END *******\n"
        with open(fout_names[typeOfData], 'w') as fout:
            fout.write("------ {} ------\n".format(score))
            fout.write(output)

        # Output prediction
        print "Making prediction on test data...\n"
        output_submission(typeOfData, clf, 
                predict_names[typeOfData]+"_"+score+".csv")
