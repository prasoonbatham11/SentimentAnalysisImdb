import os
import pandas as pd
from numpy.random import shuffle

train_data_pos = os.listdir("./aclImdb/train/pos")
train_data_neg = os.listdir("./aclImdb/train/neg")
test_data_pos = os.listdir("./aclImdb/test/pos")
test_data_neg = os.listdir("./aclImdb/test/neg")

train_data = []
test_data = []

for f in train_data_pos:
    fp = open("./aclImdb/train/pos/"+f)
    review = fp.read()
    id = f[:-4]
    train_data.append((id, 1, review))

for f in train_data_neg:
    fp = open("./aclImdb/train/neg/"+f)
    review = fp.read()
    id = f[:-4]
    train_data.append((id, 0, review))

for f in test_data_pos:
    fp = open("./aclImdb/test/pos/"+f)
    review = fp.read()
    id = f[:-4]
    test_data.append((id, 1, review))

for f in test_data_neg:
    fp = open("./aclImdb/test/neg/"+f)
    review = fp.read()
    id = f[:-4]
    test_data.append((id, 0, review))

shuffle(train_data)
shuffle(test_data)


train_df = pd.DataFrame(train_data, columns=['id', 'sentiment', 'review'])
train_df.to_csv("./train.csv", index=False)

test_df = pd.DataFrame(test_data, columns=['id', 'sentiment', 'review'])
test_df.to_csv("./test.csv", index=False)


