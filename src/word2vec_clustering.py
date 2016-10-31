from sklearn.cluster import KMeans
import numpy as np
import gensim
import time, re
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import pandas as pd


train = pd.read_csv( "../data/labeledTrainData.tsv", delimiter="\t", quoting=3 )
test = pd.read_csv( "../data/testData.tsv", delimiter="\t", quoting=3 )

def create_bag_centroids(wordlist, wordlist_centroid_map):
    num_centroid = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroid, dtype="float32")
    for word in wordlist:
        if word in wordlist_centroid_map:
            index = wordlist_centroid_map[word]

    return bag_of_centroids

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


if __name__ == '__main__':

    # start time
    start = time.time()
    model = gensim.models.Word2Vec.load('300features_40minwords_10context')
    word_vector =  model.syn0
    num_clusters = int(word_vector.shape[0] / 5)

    kmeans_clustering = KMeans(n_clusters = num_clusters)
    idx = kmeans_clustering.fit_predict(word_vector)

    # end time
    end = time.time()
    diff = end - start

    print ("Time taken for K Means clustering: ", diff, "seconds.")


    word_centroid_map = dict(zip(model.index2word, idx))

    # Contents of the first 10 cluster - can be random
    for cluster in range(0,10):
        print ("\nCluster %d" % cluster)
        words = []
        val = list(word_centroid_map.values())
        key = list(word_centroid_map.keys())
        for i in range(0,len(val)):
            if (val[i] == cluster):
                words.append(key[i])
            print (words)


    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    counter = 0

    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))


    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_centroids(review, word_centroid_map)
        counter += 1

    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    counter = 0

    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_centroids(review, word_centroid_map)
        counter += 1


    forest = RandomForestClassifier(n_estimators=100)
    print ("Fitting a random forest to labelled train data...")
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("../save/BagOfCentroids.csv", index=False, quoting=3)