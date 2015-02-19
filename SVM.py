from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

from CONFIG import *
from common import *

fout_names = {'raw':'result/SVM_raw.csv', 'tf_idf':'result/SVM_tf_idf.csv'}
predict_names = {'raw':'predict/SVM_raw.csv', 'tf_idf':'predict/SVM_tf_idf.csv'}

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
        print ("Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(SVC(), param_grid_SVM, cv=K, scoring=score,
                n_jobs=3)
        
        output = "Best parameters set found on development set: \n"
        output += clf.best_estimator_
        output += "\nGrid scores on development set: \n"
        for params, mean_score, grid_scores in clf.gird_scores_:
            output += ("%0.3f (+/- %0.03f) for %r\n" % 
            (mean_score, grid_scores.std() / 2, params))

        clf = clf.fit(X_train, Y_train)
        output += "\nDetailed Classification report:\n"
        output += "The model is trained on the full development set.\n"
        output += "The scores are computed on the full evaluation set.\n"
        y_true, y_pred = Y_test, clf.predict(X_test)
        output += classification_report(y_true, y_pred)

        # Print final statistics
        print output
        with open(fout_names[typeOfData], 'w') as fout:
            fout.write(output)

        # train the model
        clf = clf.fit(X, Y)

        # Output prediction
        output_submission(typeOfData, clf, predict_names[typeOfData])
