from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

model = Word2Vec.load("gensim_model_300features_40minwords_10context")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
stops = stopwords.words('english')
index2word = model.wv.index2word
words_per_cluster = 500
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0]//words_per_cluster

def review_to_words(review):
    review_text = BeautifulSoup(review, 'lxml').get_text()
    review_text = re.sub('[^a-zA-Z0-9]', ' ', review_text)
    words = review_text.lower().split()
    meaningful_words = [w for w in words if not w in stops]
    #final_review = ' '.join(meaningful_words)
    return meaningful_words

def clean_reviews(data):
    cleaned_reviews = []
    size = data.size
    for i in range(0, size):
        cleaned_reviews.append(review_to_words(data[i]))
        if i % 1000 == 0:
            print("Running on iteration ", i)
    return cleaned_reviews


def kmeans_clustering():

    km = KMeans(n_clusters=num_clusters)

    t = time.time()
    idx = km.fit_predict(word_vectors)
    t = time.time()-t

    print('Clustering took: ', t, 's')
    return idx

def create_bag_of_centeroids(wordlist):
    num_centeroids = max(word_centeroid_map.values())+1

    # bag_of_centeroid is similar to bag_of_words except that in bag of words we took frequency of each words
    # and here we take frequency of each cluster to create a feature vector

    bag_of_centeroid = np.zeros(num_centeroids, dtype='float32')
    for word in wordlist:
        if word in word_centeroid_map:
            index = word_centeroid_map[word] # find cluster number of word
            bag_of_centeroid[index]+=1
    return bag_of_centeroid

print('Cleaning Training Data...')
cleaned_train_reviews = clean_reviews(train['review'])
print('Cleaning Testing Data...')
cleaned_test_reviews = clean_reviews(test['review'])


idx = kmeans_clustering() # returns index of clusters for each datapoint as a numpy array size = num. of (datapoints,)

word_centeroid_map = dict(zip(index2word, idx))

train_centeroids = np.zeros((train['review'].size, num_clusters), dtype='float32')
test_centeroids = np.zeros((test['review'].size, num_clusters), dtype='float32')

c = 0
for r in cleaned_train_reviews:
    train_centeroids[c] = create_bag_of_centeroids(r)
    if c%1000==0:
        print('Creating centeroid Train Feature', c)
    c+=1

c = 0
for r in cleaned_test_reviews:
    test_centeroids[c] = create_bag_of_centeroids(r)
    if c%1000==0:
        print('Creating centeroid Test Feature', c)
    c+=1


classifier = RandomForestClassifier(n_estimators=100, min_samples_split=2)
print('Training Random Forest Classifier...')
t = time.time()
classifier.fit(train_centeroids, train['sentiment'])
t = time.time()-t
print('Classifier took ', t, 's')

print('Predicting sentiments...')
train_pred = classifier.predict(train_centeroids)
test_pred = classifier.predict(test_centeroids)
train_acc = round(accuracy_score(train_pred, train['sentiment'])*100, 2)
test_acc = round(accuracy_score(test_pred, test['sentiment'])*100,2)
print('Train Accuracy: ', train_acc, '%')
print('Test Accuracy: ', test_acc, '%')
