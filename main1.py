import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
# test = pd.read_csv("./testData.tsv", header=0, delimiter='\t', quoting=3)
stops = set(stopwords.words("english"))
ps = PorterStemmer()
lemm = WordNetLemmatizer()

def review_to_words(review):
    review_text = BeautifulSoup(review, "lxml")
    letters_only = re.sub("[^a-zA-Z0-9]", " ", review_text.get_text())
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if not w in stops] # Removing Stopwords
    #meaningful_words = [ps.stem(w) for w in meaningful_words] # Stemming
    meaningful_words = [lemm.lemmatize(w) for w in meaningful_words] # Lemmatizing
    return " ".join(meaningful_words)

def clean_reviews(data):
    cleaned_reviews = []
    size = data.size
    for i in range(0, size):
        cleaned_reviews.append(review_to_words(data[i]))
        if i%1000==0:
            print("Running on iteration ", i)
    return cleaned_reviews

def bag_of_words(max_features=5000, n_estimators=100, min_samples_split=40, criterion='entropy'):

    print("Creating Count Vectorizer and fitting cleaned train reviews...")


    vect = CountVectorizer(analyzer='word',tokenizer=None, preprocessor=None, stop_words=None, max_features=max_features)
    train_data_features = vect.fit_transform(cleaned_train_reviews).toarray()

    classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
    
    print("Training Random forest Classifier...")
    t = time.time()
    classifier.fit(train_data_features, train['sentiment'])
    t = time.time()-t
    print("Classifier took: ", t, "s")

    print("Create bag of words for test data...")
    test_data_features = vect.transform(cleaned_test_reviews)
    test_data_features = test_data_features.toarray()

    print("Predicting sentiments...")

    
    pred_train = classifier.predict(train_data_features)
    pred_test = classifier.predict(test_data_features)


    #print("making submission file...")
    #submission = pd.DataFrame({'id': test['id'], 'sentiment': pred_test})
    #submission.to_csv('submission.csv', index=False, quoting=3)

    acc_train = round(accuracy_score(pred_train, train['sentiment']) * 100, 2)
    acc_test = round(accuracy_score(pred_test, test['sentiment']) * 100, 2)
    
    print(max_features, criterion, min_samples_split, n_estimators)
    print("Training Accuracy: ", acc_train, "%")
    print("Testing Accuracy: ", acc_test, "%")

    print("\n\n\n")


print("Cleaning Train Reviews...")
cleaned_train_reviews = clean_reviews(train['review'])
print("Cleaning test reviews...")
cleaned_test_reviews = clean_reviews(test['review'])

bag_of_words(n_estimators=100, min_samples_split=2)

'''
n_estim = [100, 500, 1000]
min_split = [100, 40, 2]
for n in n_estim:
    for m in min_split:
        bag_of_words(n_estimators=n, min_samples_split=m)
'''