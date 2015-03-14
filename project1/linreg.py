from sklearn import linear_model
from sklearn import preprocessing
import csv
import numpy as np

NUM_TRAININGS = 4000
NUM_TESTINGS = 9868
f_train = 'kaggle_train_wc.csv'
f_test = 'kaggle_test_wc.csv'
data_dir = 'data/'

# Model: Linear Classification
# Train with LinearRegression, use threshold = 0.5 to classify
class LinearClassification(linear_model.LinearRegression):
	def predict(self, X):
		p = linear_model.LinearRegression.predict(self, X)
		binarizer = preprocessing.Binarizer(threshold=0.5).fit(p)
		return binarizer.transform(p)

# Load dataset
with open(data_dir + f_train, 'r') as fin:
	data = np.array(list(csv.reader(fin)))

X_train = data[1:NUM_TRAININGS+1, 1:-1].astype(int)
Y_train = data[1:NUM_TRAININGS+1, -1].astype(int)

with open(data_dir + f_test, 'r') as fin:
	data = np.array(list(csv.reader(fin)))

X_test = data[1:NUM_TESTINGS+1, 1:].astype(int)

clf = LinearClassification()

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
