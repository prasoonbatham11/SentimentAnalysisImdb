import os
import pandas as pd

unlabeled_data_pos = os.listdir('./aclImdb/train/unsup')

unlabeled_data = []
i = 0
for f in unlabeled_data_pos:
    fp = open("./aclImdb/train/unsup/"+f)
    review = fp.read()
    id = f[:-4]
    unlabeled_data.append((id, review))
    if i%1000==0:
        print('iteration', i)
    i+=1

unlabeled_df = pd.DataFrame(unlabeled_data, columns=['id', 'review'])
unlabeled_df.to_csv("./unlabeled.csv", index=False)