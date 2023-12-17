import pandas as pd


bestsellers_array = []

for chunk in pd.read_csv('train.csv', chunksize = 500000):    
    chunk.drop(columns=chunk.columns[0], axis=1, inplace=True)

    bestsellers_current= chunk.loc[chunk['isBestSeller'] == True]
    bestsellers_array.append(bestsellers_current)


# back to df, 4812 items
bestsellers = pd.concat(bestsellers_array)


bestsellers.to_csv('bestsellers.csv', sep=',', header=True)

