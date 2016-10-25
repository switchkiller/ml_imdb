import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../save/trainDataFeatures.tsv', sep='\t', index_col = 0)
df_test = pd.read_csv('../save/testDataFeatures.tsv', sep = '\t', index_col=0)
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

predictions = clf.predict(X)
submissions = pd.DataFrame({'id':df_test.id, "sentiment":predictions})
print(submissions)
submissions.to_csv('../output/submissions_kaggle.tsv', index=False, sep='\t')
# mean = np.mean(clf.predict(np.asarray(df[columns])) == df.sentiment)
# print (mean)
