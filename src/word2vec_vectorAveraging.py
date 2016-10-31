# Attemp using words to paragraph : (Features) Vector averaging method 
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import gensim, re
from nltk.corpus import stopwords

train = pd.read_csv( "../data/labeledTrainData.tsv", delimiter="\t", quoting=3 )
test = pd.read_csv( "../data/testData.tsv", delimiter="\t", quoting=3 )

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeaturesVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter%1000 == 0:
            print ("Review %d of %d" % (counter, len(review)))
            reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
            counter += 1
    return reviewFeatureVecs


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


model = gensim.models.Word2Vec.load('300features_40minwords_10context')
num_features = 300

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeaturesVecs(clean_train_reviews, model, num_features)

print ("Creating average feature vecs for the test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeaturesVecs(clean_test_reviews, model, num_features)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

print ("Fitting a random forest to labelled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# test and extract results
result = forest.predict(testDataVecs)

# Write and test results
output = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
output.to_csv("../output/Word2Vec_AvgVectors.csv", index=False, quoting=3)
