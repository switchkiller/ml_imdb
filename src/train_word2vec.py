import pandas as pd
from nltk.tokenize import word_tokenize
from feature_extraction_word2vec import review_to_words
train = pd.read_csv('../data/labeledTrainData.tsv', sep='\t')
test = pd.read_csv('../data/testData.tsv', sep='\t')
unlabeled_train = pd.read_csv( "../data/unlabeledTrainData.tsv", sep="\t", quoting=3 )
# unlabeled_train = pd.read_csv('../data/unlabeledTrainData.tsv', sep='\t')

def review_to_sentences(review, remove_stopword=False):
    raw_sentences = word_tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if (len(raw_sentence) > 0):
            sentences.append(review_to_words(raw_sentence))
    return sentences

sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review)

print (len(sentences))
