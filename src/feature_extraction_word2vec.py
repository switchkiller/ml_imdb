import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
import pickle

train_data = pd.read_csv('../data/labeledTrainData.tsv', sep='\t')
test_data = pd.read_csv('../data/testData.tsv', sep='\t')


def review_to_words(raw_review):
    prop = BeautifulSoup(raw_review).get_text()
    letters = re.sub("[^a-zA-Z]", " ", prop)
    lower_case = letters.lower()
    words = lower_case.split()
    # Dealing with the stop words
    words = [w for w in words if not w in stopwords.words("english")]
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(i) for i in words]
    return (" ".join(words))

# clean_review = review_to_words(train_data['review'][0])
# print (clean_review)

# Clean complete data
num_reviews = train_data['review'].size
clean_train_reviews = []

# print (num_reviews)
for i in range(0,num_reviews):
    print (i+1)
    clean_train_reviews.append(review_to_words(train_data['review'][i]))

with open('../save/clean_train_reviews.pickle','wb') as f:
    pickle.dump(clean_train_reviews, f)
