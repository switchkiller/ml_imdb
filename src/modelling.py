import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../save/trainDataFeatures.tsv', sep='\t', index_col = 0)
test_df = pd.read_csv('../save/testData.tsv', sep='\t', index_col=0)
# df.drop(['review'], axis=1, inplace=True)
columns = df.columns[3:]
print (columns)

X = np.asarray(df[columns])
y = np.asarray(df.sentiment.transpose())

# Using Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,y)

clf.predict(np.asarray(df[columns]))

mean = np.mean(clf.predict(np.asarray(df[columns])) == df.sentiment)
print (mean)
