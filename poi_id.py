
# coding: utf-8

# %load poi_id.py
#!/usr/bin/python

from __future__ import division # to force division results as floats
import sys
import pickle
import math 
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
                 'salary', 
                 'total_stock_value', 
                 'exercised_stock_options', 
                 'to_poi_ratio', 
                 'bonus', 
                 'total_payments', 
                 'shared_receipt_with_poi_ratio', 
                 'shared_receipt_with_poi', 
                 'from_poi_to_this_person', 
                 'from_poi_ratio', 
                 'from_this_person_to_poi'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers and fix errors

# remove two lines from the dataset that don't refer to an employee (a person)
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['TOTAL']

# fix two errors introduced in pre-processing 
# (the values for BELFER ROBERT and BHATNAGAR SANJAY have been shifted to the right)
data_dict['BELFER ROBERT']['deferral_payments'] = 'NaN'
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'
data_dict['BELFER ROBERT']['exercised_stock_options'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['deferral_payments'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['other'] = 'NaN'
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864



### Task 3: Create new feature(s)

for key in data_dict:
    # assign new variables to the dict values to make the code more readable and concise
    from_messages = data_dict[key]['from_messages']
    to_messages = data_dict[key]['to_messages']
    from_this_person_to_poi = data_dict[key]['from_this_person_to_poi']
    from_poi_to_this_person = data_dict[key]['from_poi_to_this_person']
    shared_receipt_with_poi = data_dict[key]['shared_receipt_with_poi']
    restricted_stock = data_dict[key]['restricted_stock']
    total_payments = data_dict[key]['total_payments']
    salary = data_dict[key]['salary']
    bonus = data_dict[key]['bonus']
    
    # initialise new features
    # email
    data_dict[key]['to_poi_ratio'] = 'NaN'
    data_dict[key]['from_poi_ratio'] = 'NaN'
    data_dict[key]['shared_receipt_with_poi_ratio'] = 'NaN'
    
    # financial
    data_dict[key]['restricted_stock_v_total_payments'] = 'NaN'
    data_dict[key]['restricted_stock_v_salary'] = 'NaN'
    data_dict[key]['salary_v_bonus'] = 'NaN'
    
    # create new email features
    if from_this_person_to_poi != 'NaN' and 'from_messages' != 'NaN':
        data_dict[key]['to_poi_ratio'] = from_this_person_to_poi / from_messages
        
    if to_messages != 'NaN':
        if from_poi_to_this_person != 'NaN':
            data_dict[key]['from_poi_ratio'] = from_poi_to_this_person / to_messages
    
        if shared_receipt_with_poi != 'NaN': 
            data_dict[key]['shared_receipt_with_poi_ratio'] = shared_receipt_with_poi / to_messages

    # creat new financial features
    if restricted_stock != 'NaN':
        if total_payments != 'NaN':
            data_dict[key]['restricted_stock_v_total_payments'] = restricted_stock / total_payments
    
        if salary != 'NaN':
            data_dict[key]['restricted_stock_v_salary'] = math.log10(restricted_stock / salary) 
    
    if salary != 'NaN' and bonus != 'NaN':
        data_dict[key]['salary_v_bonus'] = salary / bonus

### Store to my_dataset for easy export below.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

from sklearn.cross_validation import train_test_split

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size = 0.3, random_state = 7) 

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import tree
from sklearn import pipeline
from sklearn import grid_search
from sklearn import cross_validation 
from sklearn import preprocessing
# from sklearn import svm 
from sklearn import decomposition
# from sklearn import naive_bayes
# from sklearn import cluster


# feature scaling
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) 

# feature selection
pca = decomposition.PCA()

# classifier - algorithm of choice
algorithm = tree.DecisionTreeClassifier(random_state = 8, presort = True)
# algorithm = svm.SVC()
# algorithm = naive_bayes.GaussianNB()
# algorithm = cluster.KMeans(n_clusters = 2)

# setting up pipeline

steps = [('scale', scaler),     
         ('decompose', pca),
         ('algorithm', algorithm)]

pipeline = pipeline.Pipeline(steps)

# grid search CV

# parameters to seach through

# k means clusters
# parameters = dict(decompose__pca__n_components = [5, 10, 'mle', None],
#                   algorithm__max_iter = [20, 60, 100],
#                   algorithm__n_init = [10, 20, 30],
#                   algorithm__init = ['k-means++', 'random'])

# SVM
# parameters = dict(decompose__n_components = [5, 10, 'mle', None],
#                   algorithm__C = [0.1, 1.0, 10.0, 100.0, 1000.0],
#                   algorithm__kernel = ['linear', 'poly', 'rbf', 'sigmoid'])

# Naive bayes
# parameters = dict(decompose__n_components = [5, 10, 'mle', None])


# decision tree

parameters = dict(decompose__n_components = list(range(2, 10)),     
                  algorithm__criterion = ['entropy', 'gini'],
                  algorithm__min_samples_split = list(range(5, 15)),
                  algorithm__max_depth = [2]) 


# cross-validation
validator = cross_validation.StratifiedShuffleSplit(labels_train, n_iter = 10, test_size = 0.3, random_state = 8)

cv = grid_search.GridSearchCV(pipeline, param_grid = parameters, cv = validator, scoring = 'recall')

cv.fit(features_train, labels_train)

# get the best classifier from grid search cv

pca = decomposition.PCA(n_components = cv.best_params_['decompose__n_components'])

algorithm = tree.DecisionTreeClassifier(criterion = cv.best_params_['algorithm__criterion'], 
                                        max_depth = cv.best_params_['algorithm__max_depth'], 
                                        min_samples_split = cv.best_params_['algorithm__min_samples_split'],
                                        random_state = 8)

steps = [('scale', scaler),
         ('select_features', pca),
         ('algorithm', algorithm)]

from sklearn import pipeline

clf = pipeline.Pipeline(steps)

from tester import dump_classifier_and_data, test_classifier, load_classifier_and_data

# local testing 
# test_classifier(clf, data_dict, features_list)

# cv.best_params_

# print out Explained variance ratio for PCA
# variance = clf.named_steps['select_features'].explained_variance_ratio_

# print "Explained variance ratio:"
# for f in range(len(variance)):
#     print("%d. feature %d: %f" % (f + 1, f, variance[f]))
# print ""
    
# # print out feature importances for Decision tree
# importances = clf.named_steps['algorithm'].feature_importances_
# indices = np.argsort(importances)[::-1]

# print "Features importances:"
# for f in range(len(indices)):
#     print("%d. feature %d %f" % (f + 1, indices[f], importances[indices[f]]))

# Dump classifier, dataset, and features_list for external testing 

dump_classifier_and_data(clf, my_dataset, features_list)

