import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
df = pd.read_csv('../save/trainDataFeatures.tsv', sep='\t', index_col = 0)
clean_train_reviews = pickle.load(open('../save/clean_train_reviews.pickle', 'rb'))
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print (train_data_features.shape)

vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis = 0)
print (vocab)


# Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(n_estimators=100)
rforest = rforest.fit(train_data_features, df["sentiment"])

X = np.asarray(df['sentiment'])

predict = rforest.predict(X)
print(predict)
