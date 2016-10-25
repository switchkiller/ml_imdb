import pandas as pd
import numpy as np
df = pd.read_csv('../save/trainDataFeatures.tsv', sep='\t', index_col=0)
columns = df.columns[3:]
X = np.asarray(df[columns])
y = np.asarray(df.sentiment.transpose())

# Hold Out Method of Validation

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
# print (X_train.shape, y_train.shape)

# Using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
# print("{0} / {1} correct.".format(np.sum(y_test == y_pred), len(y_test)))

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))

# Cross Validation

from sklearn.cross_validation import cross_val_score, cross_val_predict
cv = cross_val_score(MultinomialNB(), X_train, y_train, cv=10)
m = cv.mean()
print (m)

# Confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
print (roc_auc_score(y_test,y_pred))
