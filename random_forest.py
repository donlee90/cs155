import csv
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

NUM_TRAININGS = 4000
NUM_TESTINGS = 9868
f_train = 'kaggle_train_wc.csv'
f_test = 'kaggle_test_wc.csv'
data_dir = 'data/'

# Load dataset
with open(data_dir + f_train, 'r') as fin:
	data = np.array(list(csv.reader(fin)))

X_train = data[1:NUM_TRAININGS+1, 1:-1].astype(int)
Y_train = data[1:NUM_TRAININGS+1, -1].astype(int)

with open(data_dir + f_test, 'r') as fin:
	data = np.array(list(csv.reader(fin)))

X_test = data[1:NUM_TESTINGS+1, 1:].astype(int)

clf = RandomForestClassifier(n_estimators=400)

# Cross Validation
from sklearn import cross_validation
K = 5
scores = cross_validation.cross_val_score(clf, X_train, Y_train,\
				cv=K, scoring='accuracy')
avg_score = sum(scores) / len(scores)
print('Scores = {}'.format(scores))
print('avg_score = {}'.format(avg_score))

# Output
clf = clf.fit(X_train, Y_train)
G_test = clf.predict(X_test)

with open('sub.csv', 'w') as fout:
	fout.write("Id,Prediction\n")
	id_num = 0
	for prediction in G_test:
		id_num += 1
		fout.write("%d,%d\n" % (id_num, prediction))
