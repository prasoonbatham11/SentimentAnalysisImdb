import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import sent_tokenize
import logging
from gensim.models import word2vec

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
unlabeled_train = pd.read_csv('unlabeled.csv')

def review_to_wordlist(review):
    # remove html
    review_text = BeautifulSoup(review, 'lxml').get_text()

    # Remove non alphanumeric
    review_text = re.sub('[^a-zA-Z0-9]', ' ', review_text)

    # Convert to lower case and split into words
    words = review_text.lower().split()

    return words

def review_to_sentences(review):
    raw_sentences = sent_tokenize(review)
    sentences = []
    for s in raw_sentences:
        if(len(s)>0):
            sentences.append(review_to_wordlist(s))
    return sentences

def get_sentences_from_train_data():
    sentences = []
    i = 0
    for review in train['review']:
        sentences+=review_to_sentences(review)
        if i%1000==0:
            print("Parsing review ", i, "of labeled train data")
        i+=1
    i = 0
    for review in unlabeled_train['review']:
        sentences+=review_to_sentences(review)
        if i%1000==0:
            print("Parsing review", i, "of unlabeled train data")
        i+=1
    return sentences


def train_model(num_features=300, min_word_count=40, num_workers=4, context=10, downsampling=1e-3):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(replace=True)
    model.save("gensim_model_300features_40minwords_10context")


sentences = get_sentences_from_train_data()
train_model()