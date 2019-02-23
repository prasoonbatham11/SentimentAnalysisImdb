from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

model = Word2Vec.load("gensim_model_300features_40minwords_10context")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
stops = stopwords.words('english')
index2word_set = model.wv.index2word

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


def make_feature_vectors(words, num_features=300):
    feature_vec = np.zeros((num_features,), dtype='float32')
    nwords = 0
    for w in words:
        if w in index2word_set:
            nwords+=1
            feature_vec = np.add(feature_vec, model[w])
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def get_feature_vecs_reviews(reviews, num_features=300):
    counter = 0
    review_features_vec = np.zeros((len(reviews), num_features), dtype='float32')

    for r in reviews:
        if counter%1000==0:
            print("On review ", counter)

        review_features_vec[counter] = make_feature_vectors(r, num_features)
        counter+=1
    return review_features_vec

def generate_feature_vectors():
    print('Cleaning Training Data...')
    cleaned_train_reviews = clean_reviews(train['review'])
    print('Cleaning Testing Data...')
    cleaned_test_reviews = clean_reviews(test['review'])

    print('Generating Training Vectors...')
    train_data_vectors = get_feature_vecs_reviews(cleaned_train_reviews)
    print('Generating Testing Vectors...')
    test_data_vectors = get_feature_vecs_reviews(cleaned_test_reviews)
    return train_data_vectors, test_data_vectors

def train_model(n_estimators = 100, min_samples_split=2):
    train_data_vectors, test_data_vectors = generate_feature_vectors()
    classifier = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split)
    print('Training Random Forest Classifier...')
    t = time.time()
    classifier.fit(train_data_vectors, train['sentiment'])
    t = time.time()-t
    print('Classifier took ', t, 's')

    print('Predicting sentiments...')
    train_pred = classifier.predict(train_data_vectors)
    test_pred = classifier.predict(test_data_vectors)
    train_acc = round(accuracy_score(train_pred, train['sentiment'])*100, 2)
    test_acc = round(accuracy_score(test_pred, test['sentiment'])*100,2)
    print('Train Accuracy: ', train_acc, '%')
    print('Test Accuracy: ', test_acc, '%')

def train_neural_model(hidden_layer=(5,2), max_iter=200):
    train_data_vectors, test_data_vectors = generate_feature_vectors()

    print('Scaling Features')
    scaler = StandardScaler()
    scaler.fit_transform(train_data_vectors)
    train_data_vectors = scaler.transform(train_data_vectors)
    test_data_vectors = scaler.transform(test_data_vectors)

    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer, max_iter=max_iter)
    print('Training MLP Classifier...')
    t = time.time()
    classifier.fit(train_data_vectors, train['sentiment'])
    t = time.time() - t
    print('Classifier took ', t, 's')

    print('Predicting sentiments...')
    train_pred = classifier.predict(train_data_vectors)
    test_pred = classifier.predict(test_data_vectors)
    train_acc = round(accuracy_score(train_pred, train['sentiment']) * 100, 2)
    test_acc = round(accuracy_score(test_pred, test['sentiment']) * 100, 2)
    print('Train Accuracy: ', train_acc, '%')
    print('Test Accuracy: ', test_acc, '%')

#train_model()

t = time.time()
train_neural_model((12500, 6250, 3125, 1500, 800, 200, 50), 500)
t = time.time()-t
print('Total Process took', round(t/60, 2), 'min')