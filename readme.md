
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

In the end, my features_list contained 16 features. I then run these 16 features through PCA (as part of my pipeline) to further reduce their dimension.

### Testing (new) features

The newly created features - taken together as a whole - have significant impact on the model performance. Taking only the initial 10 features and using the same tuning procedure, the model performed poorly: although precision ranged well over 0.6, recall rarelly exceeded 0.17.

How important the individual features are and how much they improve (or indeed contravene) the resulting model required additional thorough testing. Using all 16 features, the model had solid performance of around 0.3 - 0.4 for both precision and recall. However, they were two problems:  (1) My gridSearch CV didn't produce consistent results and the resulting algorithm didn't perform consistently (the performance could jump from 0.25 to 0.45, depending on the particular run of my gridsearchCV), (2) the feature importances produced by my Decision Tree algorithm were zero for all but three features.

This suggested that some of the features were introducing to the model random noise, rather than improving its performance. I decided to cut down on both the number of features and the number of components to keep after PCA (the best performing model I could get at this stage had PCA'a n_components = 10, even though the Decision Tree algorithm effectively used only three - the importance of the other components was zero).

After a second round of testing, I ended up with eleven features: the three new financial features  (restricted_stock_v_total_payments, restricted_stock_v_salary, salary_v_bonus) actually reduced the overall performance of the model as did the email features - to_messages and from_messages. Those five features thus didn't fulfill the expectation invested in them and had to leave the elite club of the final features.

## Algorithm selection 

I tested four algorithms: DecisionTree, Naive Bayes, SVM and k-means clusters. Of the four models, DecisionTree and Naive Bayes produced precision and recall over 0.3. In the end I've decided to go with DecisionTree as it performed slightly better than Naive Bayes in the early stages of the algorithm tunning.


## Algorithm tuning

To tune an algorithm is to find the best possible parametres for a given model and data. Different parametres for the same algorithm often produce very different models with different performance. It is therefore important to tune the algorithm to our needs and adjust it to our data (while avoiding overfitting to training data).

I used pipelines and GridSearchCV to find the best possible parametres for feature proprocessing (Scaling), feature transformation (PCA), and for the DecisionTree algorithm. I experimented with the different methods for scaling and with different number of components to keep after PCA. 

### Scaling
As the range of values of the different features ranged widely (thousands of dollars, number of emails, ratios), I used feature scaling to normalise the data before feeding them into PCA. In the end I opted for MinMaxScaler as it produced the best overall results. Besides MinMaxScaler, I also tried StandardScaler but the results were quite abysmal, with recall hardly exceeding 0.2.

It's good to point out that the explained variance reported by PCA didn't change when running the PCA algoritm without previously scalling the features. However, the perfomance of the model without scaling went down (though it still worked quite decently - over 0.4 for both precision and recall).

### PCA 
The model performed the best with PCA's n_components = 2. It's interesting to note that as the n_components went up, the performance was declining, until it got to 9 or 10. At this point the performance reached only a slightly worse results than for n = 2. However, as the features importances obtained from the Decision Tree algorithm were zero for all but three features, keeping n_components = 2 seemed the most sensible option.

The parametres of my algorithm:

PCA:
```python  
n_components = 2
```

DecisionTree:
```python  
  criterion =  'entropy',
  max_depth = 2,
  min_samples_split = 14,
```

Explained variance ratio for PCA:
```
1. feature 0: 0.537905
2. feature 1: 0.174883
```

Feature importances for Decision Tree algorithm:
```
1. feature 0 0.682892
2. feature 1 0.317108
```

## Validation

By validating an algorithm, we are assessing the algorithm performance beyond the initial training data. It's crucial to validate an algorithm using a new data - data that we have not used to train the algorithm. Using the same data for traing and testing typically results in overfitting the algorithm. An overfitted algorithm follows the random noise in the trainig data rather than the underlying relationship and cannot be extended to new data.

I validated mu model using the test_classifier function. Given our data (lot of missing values and only 18 pois) the best validtion strategy was StratifiedShuffleSplit (with 1000 folds), in which the model is fit 1000 times, each time with different test data (hence the shuffle split). The final performance is the average performance over the 1000 folds.

The relevant metrics for our particular case are Precision and Recall. As there are only 18 pois in the dataset, the Accuracy doesn't tell us much about the model performance. If the model simply labels all employees as non-poi, it would be correct for most cases. 

On the other hand Precision and Recall focus on how sucessful a given model is in identifying pois. - Precision is the fraction of how many times the model correctly identified a poi to the  total times the model identified someone as a poi (irrespective of whether the person actually was a poi). Recall complements Precision as the fraction of how many times the model correctly identified a poi to the total number of pois in the testing dataset.

The Precision and Recall for my model is:

- Precision: 0.54334
- Recall: 0.56100

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
