import pandas as pd
import os

for chunk in pd.read_csv('amazon_articles/amz_uk_processed_data.csv', chunksize = 100000):
    #df_sports = chunk[chunk['categoryName'] == 'Sports & Outdoors']
    #print(chunk.categoryName.unique())
    #corr=chunk['stars'].corr(chunk['reviews'])
    #print(corr)
    df = chunk.sample(frac=1)
    firstPart = df.iloc[:20000]
    secondPart = df.iloc[20000:]
    #firstPart.to_csv('split_csv_pandas/chunk{}.csv'.format(i), index=False)
    if not os.path.isfile('test.csv'):
        firstPart.to_csv('test.csv', header='column_names')
        secondPart.to_csv('train.csv', header='column_names')
    else:
        firstPart.to_csv('test.csv', mode='a', header=False)
        secondPart.to_csv('train.csv', mode='a', header=False)

