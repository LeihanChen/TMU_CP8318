"""
Assignment 2: regression
Goals: introduction to pandas, sklearn, linear and logistic regression, multi-class classification.
Start early, as you will spend time searching for the proper syntax, especially when using pandas
"""

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
Hint: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
"""

# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a
# rough overview of the data at hand.
# You will notice that there are several instances (rows), of 2 features (columns).
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
data = pandas.read_csv('RegressionData.csv', header = 0, names=['X', 'y']) # 5 points
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) # 5 points
y = data['y'].values.reshape(-1,1) # 5 points
# Plot the data using a scatter plot to visualize the data
plt.scatter(X, y) # 5 points

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() # 5 points
reg.fit(X, y) # 5 points

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) # 5 points
plt.plot(X,y, c='b') # 5 points
plt.plot(X, y_pred, 'r') # 5 points
fig.canvas.draw()

# get the bias and weight
b_0 = reg.intercept_
b_1 = reg.coef_

# # Complete the following print statement (replace the blanks _____ by using a command, do not hard-code the values):
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", b_0, " and the weight b_1 is equal to ", b_1)
# 8 points

# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]]))
# 8 points

"""
PART 2: logistic regression
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected.
You have gathered data over the years that you intend to use as a training set.
Your task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""

# Open the csv file in Excel, notepad++ or any other applications to have a rough overview of the data at hand.

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv('LogisticRegressionData.csv', header = 0, names=['Score1', 'Score2', 'y']) # 2 points
# Seperate the data features (score1 and Score2) from the class attribute
X = data[['Score1', 'Score2']] # 2 points
y = data['y'] # 2 points

# Plot the data using a scatter plot to visualize the data.
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=data['y'][i], color = data['y'][i]) # 2 points
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression() # 2 points
regS.fit(X, y) # 2 points

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) # 2 points
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=y_pred[i], color = c[y_pred[i]])  # 2 points
fig.canvas.draw()
# Notice that some of the training instances are not correctly classified. These are the training errors.

"""
PART 3: Multi-class classification using logistic regression 
Not all classification algorithms can support multi-class classification (classification tasks with more than two classes).
Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification datasets 
and fit a binary classification model on each. 
Two different examples of this approach are the One-vs-Rest and One-vs-One strategies.
"""

#  One-vs-Rest method (a.k.a. One-vs-All)

# Explain below how the One-vs-Rest method works for multi-class classification # 12 points
"""
The One-vs-Rest method splits the dataset into multiple binary classification datasets and fitting a binary classification model on each. 
It creates a separate binary classification model for each class.
Specifically, it assigns all instances of the current class a value of 1 and all instances of the other classes a value of 0.
The resulting binary classification models are then used to predict the class labels of new instances.
The final prediction for a given instance is the class label that the model with the highest confidence assigns to it.
"""

# Explain below how the One-Vs-One method works for multi-class classification # 11 points
"""
The One-vs-One method works by creating a separate binary classification model for each pair of classes. 
It will create a n * (n-1) / 2 binary classification models, where n is the number of classes.
Each model will will distinguish between two classes as a pair.
The final prediction for a given instance is determined by aggregating the predictions of all individual binary classifiers.
"""


############## FOR GRADUATE STUDENTS ONLY (the students enrolled in CPS 8318) ##############
""" 
PART 4 FOR GRADUATE STUDENTS ONLY: Multi-class classification using logistic regression project.
Please note that the grade for parts 1, 2, and 3 counts for 70% of your total grade. The following
work requires you to work on a project of your own and will account for the remaining 30% of your grade.

Choose a multi-Class Classification problem with a dataset (with a reasonable size) 
from one of the following sources (other sources are also possible, e.g., Kaggle):

•	UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets.php. 

•	KDD Cup challenges, http://www.kdd.org/kdd-cup.


Download the data, read the description, and use a logistic regression approach to solve a 
classification problem as best as you can. 
Investigate how the One-vs-Rest and One-vs-One methods can help with solving your problem.
Write up a report of approximately 2 pages, double spaced, in which you briefly describe 
the dataset (e.g., the size – number of instances and number of attributes, 
what type of data, source), the problem, the approaches that you tried and the results. 
You can use any appropriate libraries. 


Marking: Part 4 accounts for 30% of your final grade. In the write-up, cite the sources of 
your data and ideas, and use your own words to express your thoughts. 
If you have to use someone else's words or close to them, use quotes and a citation.  
The citation is a number in brackets (like [1]) that refers to a similar number in the references section 
at the end of your paper or in a footnote, where the source is given as an author, title, URL or 
journal/conference/book reference. Grammar is important. 

Submit the python script (.py file(s)) with your redacted document (PDF file) on the D2L site. 
If the dataset is not in the public domain, you also need to submit the data file. 
Name your documents appropriately:
report_Firstname_LastName.pdf
script_ Firstname_LastName.py
"""
# Using logistic regression to predict multi-class classification problems with the One-vs-Rest and One-vs-One methods.
# The dataset used is Dry Bean Data Set from the UCI Machine Learning Repository.

from ucimlrepo import fetch_ucirepo
import numpy as np

# Fetch dataset
dry_bean = fetch_ucirepo(id=602)

# Data (as pandas dataframes)
X = dry_bean.data.features
y = dry_bean.data.targets
# print the number of instances of each unique values in color, color is a dataframe
print(y.value_counts())
print(X.shape, y.shape)

# Process the features, analysis the feature distribution, and perhaps remove outliers
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Draw a class label box plot to see the distribution of the class labels
sns.countplot(data=y, x="Class")
plt.title("Distribution of Dry Bean Classes")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.savefig("dry_bean_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# Standard scaling the features but keep input X as a dataframe
feature_names = X.columns
print(feature_names)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pandas.DataFrame(X, columns=feature_names)

# Violin Plot of all features in a figure
plt.figure(figsize=(10, 6))
sns.violinplot(data=X)
plt.title("Violin Plot of Dry Bean Features")
plt.xticks(range(X.shape[1]), X.columns, rotation=45)
plt.savefig("dry_bean_violin.png", dpi=300, bbox_inches="tight")
plt.show()

# Exclude some significant outliers in the dataset if any feature it is larger or smaller than 1.5 times the interquartile range, X is a multi-column dataframe
# mask = (X - X.median()).abs() > 1.5 * (X.quantile(0.75) - X.quantile(0.25))
# Exlude outliers if any feature is larger or smaller than 3 times the standard deviation
mask = abs(X - X.mean(axis=0)) > 3 * X.std(axis=0)
X = X[~mask.any(axis=1)]
y = y[~mask.any(axis=1)]
print(X.shape, y.shape)

# There are 16 features in the dataset, so check the correlation between the features
# Draw a heatmap to visualize the correlation between the features
plt.figure(figsize=(10, 6))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Map of Dry Bean Features")
plt.savefig("dry_bean_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

# # According to the correlation map, there are some features that are highly correlated, so we can use PCA or LDA for dimensionality reduction
# from sklearn.decomposition import PCA
# pca = PCA(n_components=7)
# # Fit the model with X and apply the dimensionality reduction on X
# X = pca.fit_transform(X)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# LDA can also be used for dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

y = y.to_numpy().ravel()
lda = LinearDiscriminantAnalysis(n_components=min(len(np.unique(y)) - 1, X.shape[1]))
# Fit the model with X and apply the dimensionality reduction on X
X = lda.fit_transform(X, y)
# print the LDA results
print(lda.explained_variance_ratio_)

# Using cross-validation to evaluate the model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Use smote to resample the dataset
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy="auto", random_state=0)
X, y = smote.fit_resample(X, y)
print("X:", X.shape, "y:", y.shape)

# Compare the accuracy of the two models
num_trials = 10
acc_ovr = []
acc_ovo = []

for i in range(num_trials):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    # Create a logistic regression model with the One-vs-Rest method
    log_reg_ovr = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000, solver="lbfgs", C=0.5, penalty="l2", random_state=i
        )
    )
    # Create a logistic regression model with the One-vs-One method
    log_reg_ovo = OneVsOneClassifier(
        LogisticRegression(
            max_iter=1000, solver="lbfgs", C=0.5, penalty="l2", random_state=i
        )
    )
    scores_ovr = cross_val_score(log_reg_ovr, X, y, cv=cv, scoring="accuracy")
    scores_ovo = cross_val_score(log_reg_ovo, X, y, cv=cv, scoring="accuracy")
    print("OVR Accuracy:", scores_ovr.mean())
    print("OVO Accuracy:", scores_ovo.mean())
    acc_ovr.append(scores_ovr.mean())
    acc_ovo.append(scores_ovo.mean())

# Draw a line plot to visualize the accuracy of the two models
plt.figure(figsize=(10, 6))
plt.plot(range(num_trials), acc_ovr, label="OVR")
plt.plot(range(num_trials), acc_ovo, label="OVO")
plt.xlabel("Trial")
plt.ylabel("Accuracy")
plt.title("Accuracy of OVR and OVO Models")
plt.legend()
plt.savefig("dry_bean_ovo_ovr_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()
print("OVR Accuracy:", np.mean(acc_ovr), "OVO Accuracy:", np.mean(acc_ovo))
