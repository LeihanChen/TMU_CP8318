import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Number of labels in the dataset:", len(np.unique(y)))
print("Iris dataset shape:", X.shape)

# Visualize the data in two subplots, each showing the distribution of two features 
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax = fig.add_subplot(122)
ax.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
plt.savefig('iris_dataset_scatterplot.png', dpi=300, bbox_inches='tight')
plt.show()


# Create box plot
df = pd.DataFrame(X, columns=iris.feature_names)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
# plt.title('Box Plot of Iris Dataset Features')
plt.xticks(range(4), df.columns, rotation=45)
plt.savefig('iris_dataset_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Exclude some significant outliers in the dataset
# For example, exclude any features with three times the standard deviation from the mean
mask = abs(X - X.mean(axis=0)) > 3 * X.std(axis=0)
X = X[~mask.any(axis=1)]
y = y[~mask.any(axis=1)]
print("Iris dataset shape after removing outliers:", X.shape)

# use grid search to find the best max_depth and min_samples_split for the decision tree
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for grid search
param_grid = {
    'max_depth': range(1, 15),
    'min_samples_split': range(2, 10)
}

# configure the cross-validation procedure
outer_folds = 10
cv_outer = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
# enumerate splits
outer_results = list()
outer_best_params = list()
feature_importance = np.zeros((outer_folds, X.shape[1]))
for i, (train_ix, test_ix) in enumerate(cv_outer.split(X)):
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]
    # configure the cross-validation procedure
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
    # define the model
    model = DecisionTreeClassifier(criterion='entropy', random_state=1)
    # define search
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    # evaluate the model
    acc = accuracy_score(y_test, yhat)
    # store the result
    outer_results.append(acc)
    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    outer_best_params.append(result.best_params_)
    print('best model feature importance:', best_model.feature_importances_)
    feature_importance[i, :] = best_model.feature_importances_
# summarize the estimated performance of the model
print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
# print the average feature importance for each fold
print('Feature importance:', feature_importance.mean(axis=0))

# Draw bar plot of the best parameters for each fold
width = 0.3
plt.figure(figsize=(10, 6))
for i, params in enumerate(outer_best_params):
    plt.bar(i - width/2, params['max_depth'], width, label='max_depth', color='blue')
    plt.bar(i + width/2, params['min_samples_split'], width, label='min_samples_split', color='red')
plt.xlabel('Fold')
plt.ylabel('Best Parameters')
plt.legend(['max_depth', 'min_samples_split'])
plt.savefig('iris_best_parameters_for_each_fold.png', dpi=300, bbox_inches='tight')
plt.show()

# Draw a line plot of the four feature importance for each fold
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(feature_importance[:, i], label=iris.feature_names[i])
plt.xlabel('Fold')
plt.ylabel('Feature Importance')
plt.legend()
plt.savefig('iris_feature_importance_for_each_fold.png', dpi=300, bbox_inches='tight')
plt.show()

# Draw a line plot of the accuracy of the decision tree classifier for each fold
plt.figure(figsize=(10, 6))
plt.plot(outer_results, label='Accuracy')
plt.xlabel('Fold')
plt.ylabel('Held-outAccuracy')
plt.legend()
plt.savefig('iris_accuracy_for_each_fold.png', dpi=300, bbox_inches='tight')
plt.show()

# Use nested cross-validation to evaluate the performance of the decision tree classifier
# inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
# outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)


# Nested CV with parameter optimization
# model = DecisionTreeClassifier(criterion='entropy', random_state=1)
# clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv, refit=True, scoring='accuracy')
# nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
# average_accuracy = nested_score.mean()
# print("Average accuracy of the decision tree classifier:", average_accuracy)

# use feature_importances_ attribute of the decision tree classifier
# tree_model = clf.best_estimator_
# print("Feature importance:", tree_model.feature_importances_)

