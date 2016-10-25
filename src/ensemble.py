import pandas as pd
import numpy as np
df = pd.read_csv('../save/trainDataFeatures.tsv', sep='\t', index_col=0)
columns = df.columns[3:]
X = np.asarray(df[columns])
y = np.asarray(df.sentiment.transpose())

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(min_samples_leaf=10, min_samples_split=10)

rf.fit(X=X_train, y = y_train)
y_predict = rf.predict(X_test)

from sklearn.metrics import roc_auc_score, confusion_matrix

auc = roc_auc_score(y_test, y_predict)
con = confusion_matrix(y_test, y_predict)

print (auc)
print (con)
