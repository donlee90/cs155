# Number of samples in the dataset
NUM_TRAININGS = 4000

# NUmber of samples in the test dataset
NUM_TEST = 9868

# Types of data
types = ['raw', 'tf_idf']

# Filenames of training data
fin_names = {'raw':'data/kaggle_train_wc.csv', 'tf_idf':'data/kaggle_train_tf_idf.csv'}

# Filenames of test data
ftest_names = {'raw':'data/kaggle_test_wc.csv', 'tf_idf':'data/kaggle_test_tf_idf.csv'}

# Cross Validation 
K = 5

# Type of scoring functions
scores = ['accuracy', 'precision', 'recall', 'f1']

# Decision Tree parameters
param_grid_DT = [
    {'criterion': ["gini", "entropy"],
     'max_depth': [None] + range(1, 25)}
]
param_grid_DT_2 = [
    {'criterion': ["gini", "entropy"],
     'min_samples_leaf': [None] + range(1, 25)}
]

# Random Forest
param_grid_RF = [
    {'criterion': ['gini', 'entropy'],
     'n_estimators': [1, 5, 10],
     'bootstrap': [True, False],
     'max_depth': [None] + range(1,25)}
]

# SVM parameters
param_grid_SVM = [
  {'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
   'kernel': ['linear']},
  {'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
   'kernel': ['poly'],
   'degree': [1, 2, 3, 4, 5],
   'gamma': [0.0, 1, 0.1, 0.01, 0.001, 0.0001],
   'coef0': [0.0, 1, 10, 100]},
  {'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
   'kernel': ['rbf'],
   'gamma': [0.0, 1, 0.1, 0.01, 0.001, 0.0001]},
  {'C': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
   'kernel': ['sigmoid'],
   'gamma': [0.0, 1, 0.1, 0.01, 0.001, 0.0001],
   'coef0': [0.0, 1, 10, 100]}
]

param_grid_SVM_small = [
  {'C': [1],
   'kernel': ['rbf']},
]

