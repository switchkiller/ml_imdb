import pandas as pd
train_data = pd.read_csv('../data/labeledTrainData.tsv', sep='\t')
test_data = pd.read_csv('../data/testData.tsv', sep='\t')

import matplotlib.pyplot as plt

train_data['review_length'] = train_data.review.apply(len)
test_data['review_length'] = test_data.review.apply(len)
# p = plt.hist(train_data.review_length.values)
# plt.show()

# Checking positive and negative sentiments stats
# print ('Negative sentiments:\n',train_data[train_data.sentiment == 0].describe())
# Minimum length sentence
# print (train_data[train_data.review_length==52].review.all())
# Maximum length sentence
# print (train_data[train_data.review_length==8999].review.all())

# print ('Positive sentiments:\n',train_data[train_data.sentiment == 1].describe())


# Word Extraction
from sklearn.feature_extraction.text import CountVectorizer

# count for the stop words
vocab= ['awesome', 'good', 'amazing', 'interesting', 'terrible', 'bad', 'awful','boring']
simple_vectorizer = CountVectorizer(vocabulary = vocab)
bow = simple_vectorizer.fit_transform(train_data.review).todense()
bow_test = simple_vectorizer.fit_transform(test_data.review).todense()
# print (bow)

words = list(simple_vectorizer.vocabulary_.keys())
print (words)


df = pd.DataFrame(bow, index = train_data.index, columns = words )
df_test = pd.DataFrame(bow_test, index = test_data.index, columns=words)
# print(df)

# ll = df.words.values
p = plt.hist([df.awesome, df.good, df.amazing, df.interesting, df.terrible, df.bad, df.awful, df.boring], label = words)
# plt.legend()
# plt.show()

# val = df[(df.awesome==0) & (df.terrible==0)].count()

# print(val)
df_expand = train_data.join(df)
df_expand_test = test_data.join(df_test)

# print(df_expand_test)

df_expand.to_csv('../save/trainDataFeatures.tsv', sep='\t')
df_expand_test.to_csv('../save/testDataFeatures.tsv', sep='\t')
