import pandas as pd
import numpy as np

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB(force_alpha=True)

first = True
bestsellers_array = []

df_y_test_total = []
data_y_pred = []

mean_ = 385
rev_stdev = 3827

for chunk in pd.read_csv('train.csv', chunksize = 500000):
    bestsellers_current= chunk.loc[chunk['isBestSeller'] == True]
    bestsellers_array.append(bestsellers_current)



# back to df, 4812 items
bestsellers = pd.concat(bestsellers_array)

#print(bestsellers.categoryName.unique())

# ?? make clusters of the abundant data, so 1200 per chunk
for chunk in pd.read_csv('train.csv', chunksize = 100000):
    df = chunk[['reviews', 'boughtInLastMonth', 'price', 'categoryName', 'isBestSeller']].copy()

    # get a df without bestsellers
    df_intermediate = df.loc[df['isBestSeller'] == False]

    df_no_bs = df_intermediate[['reviews', 'boughtInLastMonth', 'price', 'categoryName', 'isBestSeller']].copy()
    df_with_bs = bestsellers[['reviews', 'boughtInLastMonth', 'price', 'categoryName', 'isBestSeller']].copy()

    #df_no_bs['reviewNormalized'] = (df_no_bs['reviews'] - mean_)/rev_stdev
    #df_with_bs['reviewNormalized'] = (df_with_bs['reviews'] - mean_)/rev_stdev

    df_with_bs['categoryMedianPrice'] = 0
    df_with_bs.loc[(df_with_bs['categoryName'] == 'Boys') | (df_with_bs['categoryName'] == 'Handmade Home & Kitchen Products'
                                                        ) | (df_with_bs['categoryName'] == 'Car & Motorbike'
                                                        ) | (df_with_bs['categoryName'] == 'Hardware'
                                                        ) | (df_with_bs['categoryName'] == 'Wearable Technology'
                                                        ) | (df_with_bs['categoryName'] == 'USB Gadgets'
                                                        ) | (df_with_bs['categoryName'] == 'Light Bulbs'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Gifts'
                                                        ) | (df_with_bs['categoryName'] == 'Calendars & Personal Organisers'
                                                        ) | (df_with_bs['categoryName'] == 'Pet Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Plants, Seeds & Bulbs'
                                                        ) | (df_with_bs['categoryName'] == 'Kids\' Art & Craft Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Skin Care'
                                                        ) | (df_with_bs['categoryName'] == 'Hobbies'
                                                        ) | (df_with_bs['categoryName'] == 'Computer Screws'
                                                        ) | (df_with_bs['categoryName'] == 'Bath & Body'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Kitchen & Dining'
                                                        ) | (df_with_bs['categoryName'] == 'Agricultural Equipment & Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Power & Hand Tools'
                                                        ) | (df_with_bs['categoryName'] == 'Boating Footwear'
                                                        ) | (df_with_bs['categoryName'] == 'SIM Cards'
                                                        ) | (df_with_bs['categoryName'] == 'Cutting Tools'
                                                        ) | (df_with_bs['categoryName'] == 'Abrasive & Finishing Products'
                                                        ) | (df_with_bs['categoryName'] == 'Mobile Phone Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Cables & Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Manicure & Pedicure Products'
                                                        ) | (df_with_bs['categoryName'] == 'Rough Plumbing'
                                                        ) | (df_with_bs['categoryName'] == 'Kitchen Tools & Gadgets'
                                                        ) | (df_with_bs['categoryName'] == 'Cushions & Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Pens, Pencils & Writing Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'School & Educational Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Home Fragrance'
                                                        ) | (df_with_bs['categoryName'] == 'Games & Game Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Beauty'
                                                        ) | (df_with_bs['categoryName'] == 'Bedding Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Kitchen Linen'
                                                        ) | (df_with_bs['categoryName'] == 'Hiking Games & Game Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Artwork'
                                                        ) | (df_with_bs['categoryName'] == 'Make-up'
                                                        ) | (df_with_bs['categoryName'] == 'Industrial Electrical'
                                                        ) | (df_with_bs['categoryName'] == 'Painting Supplies, Tools & Wall Treatments'
                                                        ) | (df_with_bs['categoryName'] == 'Electrical Power Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Safety & Security'
                                                        ) | (df_with_bs['categoryName'] == 'Vacuums & Floorcare'
                                                        ) | (df_with_bs['categoryName'] == 'Radio Communication'
                                                        ) | (df_with_bs['categoryName'] == 'Girls'
                                                        ) | (df_with_bs['categoryName'] == 'Tablet Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Baby'
                                                        ) | (df_with_bs['categoryName'] == 'Office Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Baby & Toddler Toys'
                                                        ) | (df_with_bs['categoryName'] == 'Learning & Education Toys'
                                                        ) | (df_with_bs['categoryName'] == 'Health & Personal Care'
                                                        ) | (df_with_bs['categoryName'] == 'Signs & Plaques'
                                                        ) | (df_with_bs['categoryName'] == 'Gardening'
                                                        ) | (df_with_bs['categoryName'] == 'Decorative Artificial Flora'
                                                        ) | (df_with_bs['categoryName'] == 'Toy Advent Calendars'
                                                        ) | (df_with_bs['categoryName'] == 'Candles & Holders'
                                                        ) | (df_with_bs['categoryName'] == 'External Sound Cards'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Clothing, Shoes & Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Home Décor'
                                                        ) | (df_with_bs['categoryName'] == 'Professional Education Supplies'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade'
                                                        ) | (df_with_bs['categoryName'] == 'Headphones, Earphones & Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Ironing & Steamers'
                                                        ) | (df_with_bs['categoryName'] == 'Household Batteries, Chargers & Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Grocery'
                                                        ) | (df_with_bs['categoryName'] == 'Hi-Fi & Home Audio Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Bakeware'
                                                        ) | (df_with_bs['categoryName'] == 'Torches'
                                                        ) | (df_with_bs['categoryName'] == 'Electrical'
                                                        ) | (df_with_bs['categoryName'] == 'Adapters'
                                                        ) | (df_with_bs['categoryName'] == 'Computer Memory Card Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Hard Drive Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Garden Tools & Watering Equipment'
                                                        ) | (df_with_bs['categoryName'] == 'Office Paper'
                                                        ) | (df_with_bs['categoryName'] == 'Jigsaws & Puzzles'
                                                        ) | (df_with_bs['categoryName'] == 'Decorative Home Accessories'
                                                        ) | (df_with_bs['categoryName'] == 'Clocks'
                                                        ) | (df_with_bs['categoryName'] == 'Doormats'
                                                        ) | (df_with_bs['categoryName'] == 'Photo Frames'
                                                        ) | (df_with_bs['categoryName'] == 'Kids\' Dress Up & Pretend Play'
                                                        ) | (df_with_bs['categoryName'] == 'Soft Toys'
                                                        ) | (df_with_bs['categoryName'] == 'Handmade Jewellery'
                                                        ) | (df_with_bs['categoryName'] == 'Hair Care'
                                                        ) | (df_with_bs['categoryName'] == 'Arts & Crafts'
                                                        ) , 'categoryMedianPrice'] = 1
    
    df_no_bs['categoryMedianPrice'] = 0
    df_no_bs.loc[(df_no_bs['categoryName'] == 'Boys') | (df_no_bs['categoryName'] == 'Handmade Home & Kitchen Products'
                                                        ) | (df_no_bs['categoryName'] == 'Car & Motorbike'
                                                        ) | (df_no_bs['categoryName'] == 'Hardware'
                                                        ) | (df_no_bs['categoryName'] == 'Wearable Technology'
                                                        ) | (df_no_bs['categoryName'] == 'USB Gadgets'
                                                        ) | (df_no_bs['categoryName'] == 'Light Bulbs'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Gifts'
                                                        ) | (df_no_bs['categoryName'] == 'Calendars & Personal Organisers'
                                                        ) | (df_no_bs['categoryName'] == 'Pet Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Plants, Seeds & Bulbs'
                                                        ) | (df_no_bs['categoryName'] == 'Kids\' Art & Craft Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Skin Care'
                                                        ) | (df_no_bs['categoryName'] == 'Hobbies'
                                                        ) | (df_no_bs['categoryName'] == 'Computer Screws'
                                                        ) | (df_no_bs['categoryName'] == 'Bath & Body'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Kitchen & Dining'
                                                        ) | (df_no_bs['categoryName'] == 'Agricultural Equipment & Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Power & Hand Tools'
                                                        ) | (df_no_bs['categoryName'] == 'Boating Footwear'
                                                        ) | (df_no_bs['categoryName'] == 'SIM Cards'
                                                        ) | (df_no_bs['categoryName'] == 'Cutting Tools'
                                                        ) | (df_no_bs['categoryName'] == 'Abrasive & Finishing Products'
                                                        ) | (df_no_bs['categoryName'] == 'Mobile Phone Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Cables & Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Manicure & Pedicure Products'
                                                        ) | (df_no_bs['categoryName'] == 'Rough Plumbing'
                                                        ) | (df_no_bs['categoryName'] == 'Kitchen Tools & Gadgets'
                                                        ) | (df_no_bs['categoryName'] == 'Cushions & Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Pens, Pencils & Writing Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'School & Educational Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Home Fragrance'
                                                        ) | (df_no_bs['categoryName'] == 'Games & Game Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Beauty'
                                                        ) | (df_no_bs['categoryName'] == 'Bedding Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Kitchen Linen'
                                                        ) | (df_no_bs['categoryName'] == 'Hiking Games & Game Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Artwork'
                                                        ) | (df_no_bs['categoryName'] == 'Make-up'
                                                        ) | (df_no_bs['categoryName'] == 'Industrial Electrical'
                                                        ) | (df_no_bs['categoryName'] == 'Painting Supplies, Tools & Wall Treatments'
                                                        ) | (df_no_bs['categoryName'] == 'Electrical Power Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Safety & Security'
                                                        ) | (df_no_bs['categoryName'] == 'Vacuums & Floorcare'
                                                        ) | (df_no_bs['categoryName'] == 'Radio Communication'
                                                        ) | (df_no_bs['categoryName'] == 'Girls'
                                                        ) | (df_no_bs['categoryName'] == 'Tablet Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Baby'
                                                        ) | (df_no_bs['categoryName'] == 'Office Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Baby & Toddler Toys'
                                                        ) | (df_no_bs['categoryName'] == 'Learning & Education Toys'
                                                        ) | (df_no_bs['categoryName'] == 'Health & Personal Care'
                                                        ) | (df_no_bs['categoryName'] == 'Signs & Plaques'
                                                        ) | (df_no_bs['categoryName'] == 'Gardening'
                                                        ) | (df_no_bs['categoryName'] == 'Decorative Artificial Flora'
                                                        ) | (df_no_bs['categoryName'] == 'Toy Advent Calendars'
                                                        ) | (df_no_bs['categoryName'] == 'Candles & Holders'
                                                        ) | (df_no_bs['categoryName'] == 'External Sound Cards'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Clothing, Shoes & Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Home Décor'
                                                        ) | (df_no_bs['categoryName'] == 'Professional Education Supplies'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade'
                                                        ) | (df_no_bs['categoryName'] == 'Headphones, Earphones & Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Ironing & Steamers'
                                                        ) | (df_no_bs['categoryName'] == 'Household Batteries, Chargers & Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Grocery'
                                                        ) | (df_no_bs['categoryName'] == 'Hi-Fi & Home Audio Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Bakeware'
                                                        ) | (df_no_bs['categoryName'] == 'Torches'
                                                        ) | (df_no_bs['categoryName'] == 'Electrical'
                                                        ) | (df_no_bs['categoryName'] == 'Adapters'
                                                        ) | (df_no_bs['categoryName'] == 'Computer Memory Card Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Hard Drive Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Garden Tools & Watering Equipment'
                                                        ) | (df_no_bs['categoryName'] == 'Office Paper'
                                                        ) | (df_no_bs['categoryName'] == 'Jigsaws & Puzzles'
                                                        ) | (df_no_bs['categoryName'] == 'Decorative Home Accessories'
                                                        ) | (df_no_bs['categoryName'] == 'Clocks'
                                                        ) | (df_no_bs['categoryName'] == 'Doormats'
                                                        ) | (df_no_bs['categoryName'] == 'Photo Frames'
                                                        ) | (df_no_bs['categoryName'] == 'Kids\' Dress Up & Pretend Play'
                                                        ) | (df_no_bs['categoryName'] == 'Soft Toys'
                                                        ) | (df_no_bs['categoryName'] == 'Handmade Jewellery'
                                                        ) | (df_no_bs['categoryName'] == 'Hair Care'
                                                        ) | (df_no_bs['categoryName'] == 'Arts & Crafts'
                                                        ) , 'categoryMedianPrice'] = 1
    

    sample_abundant = df_no_bs[:4800]


    df_united_array = [df_with_bs, sample_abundant]
 
    df_united = pd.concat(df_united_array)

    #print(df_united['categoryName'].head(20))

    feature_cols = ['reviews', 'boughtInLastMonth', 'price', 'categoryMedianPrice']

    y = (df_united['isBestSeller'])
    X = (df_united[feature_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    df_y_test_total.append(y_test)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    #if first:
     #   logreg_1.fit(X_train, y_train)
      #  first = False

    data_y_pred.append(y_pred)


final_y_test = pd.concat(df_y_test_total)
arr = np.concatenate(data_y_pred).ravel()

cm = confusion_matrix(final_y_test, arr, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()


#
 #   kmeans = KMeans(n_clusters=100).fit(df)
  #  centroids = kmeans.cluster_centers_
   # closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df)
   # print('ok')

df_test = pd.read_csv('test.csv')

df_test_intermed = df_test[['reviews', 'boughtInLastMonth', 'price', 'categoryName', 'isBestSeller']].copy()
df_test_intermed['reviewNormalized'] = (df_test_intermed['reviews'] - mean_)/rev_stdev

df_test_intermed['categoryMedianPrice'] = 0
df_test_intermed.loc[(df_test_intermed['categoryName'] == 'Boys') | (df_test_intermed['categoryName'] == 'Handmade Home & Kitchen Products'
                                                        ) | (df_test_intermed['categoryName'] == 'Car & Motorbike'
                                                        ) | (df_test_intermed['categoryName'] == 'Hardware'
                                                        ) | (df_test_intermed['categoryName'] == 'Wearable Technology'
                                                        ) | (df_test_intermed['categoryName'] == 'USB Gadgets'
                                                        ) | (df_test_intermed['categoryName'] == 'Light Bulbs'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Gifts'
                                                        ) | (df_test_intermed['categoryName'] == 'Calendars & Personal Organisers'
                                                        ) | (df_test_intermed['categoryName'] == 'Pet Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Plants, Seeds & Bulbs'
                                                        ) | (df_test_intermed['categoryName'] == 'Kids\' Art & Craft Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Skin Care'
                                                        ) | (df_test_intermed['categoryName'] == 'Hobbies'
                                                        ) | (df_test_intermed['categoryName'] == 'Computer Screws'
                                                        ) | (df_test_intermed['categoryName'] == 'Bath & Body'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Kitchen & Dining'
                                                        ) | (df_test_intermed['categoryName'] == 'Agricultural Equipment & Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Power & Hand Tools'
                                                        ) | (df_test_intermed['categoryName'] == 'Boating Footwear'
                                                        ) | (df_test_intermed['categoryName'] == 'SIM Cards'
                                                        ) | (df_test_intermed['categoryName'] == 'Cutting Tools'
                                                        ) | (df_test_intermed['categoryName'] == 'Abrasive & Finishing Products'
                                                        ) | (df_test_intermed['categoryName'] == 'Mobile Phone Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Cables & Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Manicure & Pedicure Products'
                                                        ) | (df_test_intermed['categoryName'] == 'Rough Plumbing'
                                                        ) | (df_test_intermed['categoryName'] == 'Kitchen Tools & Gadgets'
                                                        ) | (df_test_intermed['categoryName'] == 'Cushions & Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Pens, Pencils & Writing Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'School & Educational Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Home Fragrance'
                                                        ) | (df_test_intermed['categoryName'] == 'Games & Game Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Beauty'
                                                        ) | (df_test_intermed['categoryName'] == 'Bedding Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Kitchen Linen'
                                                        ) | (df_test_intermed['categoryName'] == 'Hiking Games & Game Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Artwork'
                                                        ) | (df_test_intermed['categoryName'] == 'Make-up'
                                                        ) | (df_test_intermed['categoryName'] == 'Industrial Electrical'
                                                        ) | (df_test_intermed['categoryName'] == 'Painting Supplies, Tools & Wall Treatments'
                                                        ) | (df_test_intermed['categoryName'] == 'Electrical Power Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Safety & Security'
                                                        ) | (df_test_intermed['categoryName'] == 'Vacuums & Floorcare'
                                                        ) | (df_test_intermed['categoryName'] == 'Radio Communication'
                                                        ) | (df_test_intermed['categoryName'] == 'Girls'
                                                        ) | (df_test_intermed['categoryName'] == 'Tablet Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Baby'
                                                        ) | (df_test_intermed['categoryName'] == 'Office Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Baby & Toddler Toys'
                                                        ) | (df_test_intermed['categoryName'] == 'Learning & Education Toys'
                                                        ) | (df_test_intermed['categoryName'] == 'Health & Personal Care'
                                                        ) | (df_test_intermed['categoryName'] == 'Signs & Plaques'
                                                        ) | (df_test_intermed['categoryName'] == 'Gardening'
                                                        ) | (df_test_intermed['categoryName'] == 'Decorative Artificial Flora'
                                                        ) | (df_test_intermed['categoryName'] == 'Toy Advent Calendars'
                                                        ) | (df_test_intermed['categoryName'] == 'Candles & Holders'
                                                        ) | (df_test_intermed['categoryName'] == 'External Sound Cards'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Clothing, Shoes & Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Home Décor'
                                                        ) | (df_test_intermed['categoryName'] == 'Professional Education Supplies'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade'
                                                        ) | (df_test_intermed['categoryName'] == 'Headphones, Earphones & Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Ironing & Steamers'
                                                        ) | (df_test_intermed['categoryName'] == 'Household Batteries, Chargers & Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Grocery'
                                                        ) | (df_test_intermed['categoryName'] == 'Hi-Fi & Home Audio Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Bakeware'
                                                        ) | (df_test_intermed['categoryName'] == 'Torches'
                                                        ) | (df_test_intermed['categoryName'] == 'Electrical'
                                                        ) | (df_test_intermed['categoryName'] == 'Adapters'
                                                        ) | (df_test_intermed['categoryName'] == 'Computer Memory Card Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Hard Drive Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Garden Tools & Watering Equipment'
                                                        ) | (df_test_intermed['categoryName'] == 'Office Paper'
                                                        ) | (df_test_intermed['categoryName'] == 'Jigsaws & Puzzles'
                                                        ) | (df_test_intermed['categoryName'] == 'Decorative Home Accessories'
                                                        ) | (df_test_intermed['categoryName'] == 'Clocks'
                                                        ) | (df_test_intermed['categoryName'] == 'Doormats'
                                                        ) | (df_test_intermed['categoryName'] == 'Photo Frames'
                                                        ) | (df_test_intermed['categoryName'] == 'Kids\' Dress Up & Pretend Play'
                                                        ) | (df_test_intermed['categoryName'] == 'Soft Toys'
                                                        ) | (df_test_intermed['categoryName'] == 'Handmade Jewellery'
                                                        ) | (df_test_intermed['categoryName'] == 'Hair Care'
                                                        ) | (df_test_intermed['categoryName'] == 'Arts & Crafts'
                                                        ) , 'categoryMedianPrice'] = 1
    

feature_cols = ['reviews', 'boughtInLastMonth', 'price', 'categoryMedianPrice']


y_test = (df_test_intermed['isBestSeller'])
X_test = (df_test_intermed[feature_cols])

y_test_pred = clf.predict(X_test)

cm_2 = confusion_matrix(y_test, y_test_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2,
                              display_labels=clf.classes_)
disp.plot()
plt.show()