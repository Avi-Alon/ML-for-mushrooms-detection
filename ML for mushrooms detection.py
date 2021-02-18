# Classification for mushrooms detection
# Here's an application of machine learning that could save your life!
# We will be working with the UCI Mushroom Data Set
# stored in readonly/mushrooms.csv.
# The data will be used to train a model to predict whether or not a mushroom is poisonous.
# The following attributes are provided:

#Attribute Information:

# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
# 4. bruises?: bruises=t, no=f
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n
# 7. gill-spacing: close=c, crowded=w, distant=d
# 8. gill-size: broad=b, narrow=n
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p,
#   purple=u, red=e, white=w, yellow=y
# 10. stalk-shape: enlarging=e, tapering=t
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# 16. veil-type: partial=p, universal=u
# 17. veil-color: brown=n, orange=o, white=w, yellow=y
# 18. ring-number: none=n, one=o, two=t
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
#
# The data in the mushrooms dataset is currently encoded with strings.
# These values will need to be encoded to numeric to work with sklearn.
# We'll use pd.get_dummies to convert the categorical variables into indicator variables.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for phase 1
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in phases 2,3, we will create a smaller version of the
# entire mushroom dataset for use in those phases.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for phases 2,3.
X_subset = X_test2
y_subset = y_test2

# ------------------------------ Phase 1 --------------------------------

# Using X_train2 and y_train2 from the preceeding cell,
# train a DecisionTreeClassifier with default parameters and random_state=0.
# What are the 5 most important features found by the decision tree?
# As a reminder, the feature names are available in the X_train2.columns property,
# and the order of the features in X_train2.columns matches
# the order of the feature importance values in the classifier's feature_importances_ property.
# This function should return a list of length 5 containing the feature names in descending order of importance.
# Note: remember that you also need to set random_state in the DecisionTreeClassifier.

def phase_one():
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    #print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train2, y_train2)))
    #print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test2, y_test2)))
    data = np.array(clf.feature_importances_)
    index = X_train2.columns
    feature_imp_ser = pd.Series(data, index)
    sorted_feature_imp_ser = feature_imp_ser.sort_values(ascending=False)
    final_list = sorted_feature_imp_ser[0:5].index.tolist()
    return final_list
#print(phase_one())

# ------------------------------ Phase 2 --------------------------------
#For this phase, we're going to use the validation_curve function in sklearn.model_selection
# to determine training and test scores for a Support Vector Classifier (SVC)
# with varying parameter values. Recall that the validation_curve function,
# in addition to taking an initialized unfitted classifier object, takes a dataset as input
# and does its own internal train-test splits to compute results.

# Because creating a validation curve requires fitting multiple models,
# for performance reasons this phase will use just a subset of the original mushroom dataset:
# We use the variables X_subset and y_subset as input to the validation curve function
# (instead of X_mush and y_mush) to reduce computation time.

# The initialized unfitted classifier object we'll be using is a Support Vector Classifier
# with radial basis kernel. So the first step is to create an SVC object with default parameters
# (i.e. kernel='rbf', C=1) and random_state=0. Recall that the kernel width of the RBF kernel
# is controlled using the gamma parameter.

# With this classifier, and the dataset in X_subset, y_subset,
# we explore the effect of gamma on classifier accuracy by using the validation_curve function
# to find the training and test scores for 6 values of gamma from 0.0001 to 10
# (i.e. np.logspace(-4,1,6)).

# Recall that we can specify what scoring metric we want validation_curve
# to use by setting the "scoring" parameter. In this case, we want to use "accuracy" as the scoring metric.

# For each level of gamma, validation_curve will fit 3 models on different subsets of the data,
# returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.

# We will find the mean score across the three models for each level of gamma for both arrays,
# creating two arrays of length 6, and return a tuple with the two arrays.

# e.g.

# if one of your array of scores is

# array([[ 0.5,  0.4,  0.6],
#        [ 0.7,  0.8,  0.7],
#        [ 0.9,  0.8,  0.8],
#        [ 0.8,  0.7,  0.8],
#        [ 0.7,  0.6,  0.6],
#        [ 0.4,  0.6,  0.5]])
# it should then become
#
# array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
# This function should return one tuple of numpy arrays
# (training_scores, test_scores) where each array in the tuple has shape (6,).

def phase_two():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    import numpy as np
    clf = SVC(kernel='rbf', C=1)
    param_range = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(clf, X_subset, y_subset,
                                                 param_name='gamma',
                                                 param_range=param_range, cv=3, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)


    return (train_scores_mean, test_scores_mean)


#print(phase_two())

# ------------------------------ Phase 3 --------------------------------
# Based on the scores from phase 2, we can explore what gamma value corresponds to a model that is underfitting
# (and has the worst test set accuracy), what gamma value corresponds to a model
# that is overfitting (and has the worst test set accuracy) and what choice of gamma would
# be the best choice for a model with good generalization performance on this dataset
# high accuracy on both training and test set).


# This function will return one tuple with the degree values in this order:
# (Underfitting, Overfitting, Good_Generalization)

# This Auxiliary plot code based on scikit-learn validation_plot example
#  See:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
plt.figure()
train_scores_mean = phase_two()[0]
#train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = phase_two()[1]
#test_scores_std = np.std(test_scores, axis=1)

plt.title('Validation Curve with SVM')
plt.xlabel('$\gamma$ (gamma)')
plt.ylabel('Score')
plt.ylim(0.4, 1.1)
lw = 2
param_range = np.logspace(-4, 1, 6)
plt.semilogx(param_range, train_scores_mean, 'o-', label='Training score',
            color='darkorange', lw=lw)

#plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                train_scores_mean + train_scores_std, alpha=0.2,
#                color='darkorange', lw=lw)

plt.semilogx(param_range, test_scores_mean, 'o-', label='Cross-validation score',
            color='navy', lw=lw)

#plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                test_scores_mean + test_scores_std, alpha=0.2,
#                color='navy', lw=lw)

plt.legend(loc='lower center')
plt.show()


def phase_three():
     (Underfitting, Overfitting, Good_Generalization) = (0.001, 10, 0.1) # According to the plot above
     return (Underfitting, Overfitting, Good_Generalization)
print(phase_three())