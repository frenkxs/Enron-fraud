
# Project 5: Identifying Fraud from Enron Emails and Financial Data
### Premysl Velek | September 2015

## Introduction

Enron Corporation was one of the largest US companies  - in 2001 it employed approximately 20,000 staff and was one of the world's major electricity, natural gas, communications and pulp and paper companies, with claimed revenues of nearly $111 billion during 2000.

At the end of 2001, it was revealed that its reported financial condition was sustained by institutionalised, systematic, and creatively planned accounting fraud, known since as the Enron scandal. ([Wikipedia](https://en.wikipedia.org/wiki/Enron)).

The goal of this project is to use Machine Learning to identify 'persons of interest' (Enron employees who may have committed fraud), based on financial and email data of Enron's personnnel.

## Goals
The goal of the project is to develop a predictive model to identify Enron employees who may have been involved in  the corporate fraud  based on the public Enron financial and email dataset and on a manually compiled list of Enron's top executives who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. Building the model, I used supervised learning for labeled data the the sklearn python library.

The dataset provided include 146 data points (Enron employees) with 21 observations. The 21 observations are split up into financial features, email features and the POI (person of interest) label.

The full info about the dataset structure:

```python
<class 'pandas.core.frame.DataFrame'>
Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
Data columns (total 21 columns):
bonus                        82 non-null float64
deferral_payments            39 non-null float64
deferred_income              49 non-null float64
director_fees                17 non-null float64
email_address                0 non-null float64
exercised_stock_options      102 non-null float64
expenses                     95 non-null float64
from_messages                86 non-null float64
from_poi_to_this_person      86 non-null float64
from_this_person_to_poi      86 non-null float64
loan_advances                4 non-null float64
long_term_incentive          66 non-null float64
other                        93 non-null float64
poi                          146 non-null bool
restricted_stock             110 non-null float64
restricted_stock_deferred    18 non-null float64
salary                       95 non-null float64
shared_receipt_with_poi      86 non-null float64
to_messages                  86 non-null float64
total_payments               125 non-null float64
total_stock_value            126 non-null float64
dtypes: bool(1), float64(20)
memory usage: 24.1+ KB
```

The intial data exploration revealed two outliers that clearly didn't belong to the dataset:

- TOTAL (grand sum of each observation across all data points)
- THE TRAVEL AGENCY IN THE PARK (not a person)

Two additional errors in the dataset needed correction (the observations for BELFER ROBERT and BHATNAGAR SANJAY were both shifted to the left when feeding the data into the right format).

I removed the two outliers and corrected the two errors before selecting the features for my classifier.

## Feature selection

The features I selected for my classifier, resulted form an exploratory data analysis of the dataset (as documented in enron_eda.pynb).

The dataset contains a lot of missing values, so I first eliminated those features that contain very few data (in general and in particular for poi-s): loan_advances, director_fees, deferral_payments, deffered_income and long_term_incentive.

I also created new features. First of all, I looked at the ratios of the number of emails from/to poi to the total number of emails sent/received. I created three new features:
```
- to_poi_ratio = from_this_person_to_poi / from_messages      
- from_poi_ratio = from_poi_to_this_person / to_messages
- shared_receipt_with_poi_ratio = shared_receipt_with_poi / to_messages
```

I created three more financial features, based on the findings of the exploratory analysis (see enron_eda.pynb):
```
- restricted_stock_v_total_payments = restricted_stock / total_payments
- restricted_stock_v_salary = log10(restricted_stock / salary)
- salary_v_bonus = salary / bonus
```
I created the email-related features by 'thinking like a human' - simply by looking at 'what would make sense' given the data. Conversely, the financial features were driven by 'computer thinking' - by looking mostly at the disctibutions of the values and how they differ for pois and non-pois.

In the end, my intitial set of features contained 16 features. I then run these 16 features through PCA (as part of my pipeline) to further reduce their dimension.


## Algorithm selection 

I tested four algorithms: DecisionTree, Naive Bayes, SVM and k-means clusters. Of the four models, DecisionTree and Naive Bayes produced precision and recall over 0.3. In the end I've decided to go with DecisionTree as it performed slightly better than Naive Bayes in the early stages of the algorithm tunning.


## Algorithm tuning

To tune an algorithm is to find the best possible parametres for a given model and data. Different parametres for the same algorithm often produce very different models with different performance. It is therefore important to tune the algorithm to our needs and adjust it to our data (while avoiding overfitting to training data).

I used pipelines and GridSearchCV to find the best possible parametres for the DecisionTree algorithm. Besides the algorithm tuning, the pipeline also contained feature transformation (PCA). I've run the GridSearchCV several times to get an idea what parameters get picked and then selected those that performed best in the validation phase.

The parametres of my algorithm:

PCA:
```python  
n_components = 10
```
DecisionTree:
```python  
  criterion =  'entropy',
  max_depth = 2,
  min_samples_split = 12,
```

## Validation

By validating an algorithm, we are assessing the algorithm performance beyond the initial training data. It's crucial to validate an algorithm using a new data - data that we have not used to train the algorithm. Using the same data for traing and testing typically results in overfitting the algorithm. An overfitted algorithm follows the random noise in the trainig data rather than the underlying relationship and cannot be extended to new data.

I validated mu model using the test_classifier function. Given our data (lot of missing values and only 18 pois) the best validtion strategy was StratifiedShuffleSplit (with 1000 folds), in which the model is fit 1000 times, each time with different test data (hence the shuffle split). The final performance is the average performance over the 1000 folds.

The relevant metrics for our particular case are Precision and Recall. As there are only 18 pois in the dataset, the Accuracy doesn't tell us much about the model performance. If the model simply labels all employees as non-poi, it would be correct for most cases. 

On the other hand Precision and Recall focus on how sucessful a given model is in identifying pois. - Precision is the fraction of how many times the model correctly identified a poi to the total times the model identified a poi. Recall complements Precision as the fraction of how many times the model correctly identified a poi to the total number of pois in the testing dataset.

The Precision and Recall for my model is:

- Precision: 0.53185
- Recall: 0.45500

(For the purposes of reproducibility I set random_state for the initial train-test split of the dataset, as well as the cross-validation and the final classifier.)

# References:

- Workflows in Python: Using Pipeline and GridSearchCV for More Compact and Comprehensive Code
https://civisanalytics.com/blog/data-science/2016/01/06/workflows-python-using-pipeline-gridsearchcv-for-compact-code/ 


- Python feature selection in pipeline: how determine feature names
https://stackoverflow.com/questions/33376078/python-feature-selection-in-pipeline-how-determine-feature-names


- Python, sklearn: Order of Pipeline operation with MinMaxScaler and SVC
http://stackoverflow.com/questions/36584873/python-sklearn-order-of-pipeline-operation-with-minmaxscaler-and-svc


- Feature_importances
https://discussions.udacity.com/t/feature-importances-/173319/2


- Inconsistent results with StratifiedShuffleSplit
https://discussions.udacity.com/t/inconsistent-results-with-stratifiedshufflesplit/190247


- Feature selection: how to deal with highly correlated features?
https://discussions.udacity.com/t/feature-selection-how-to-deal-with-highly-correlated-features/166685/1
