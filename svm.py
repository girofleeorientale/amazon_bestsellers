import pandas as pd
import numpy as np

# import the class
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svc=SVC()

df_y_test_total = []
data_y_pred = []


for chunk in pd.read_csv('../train.csv', chunksize = 100000):
    scaler = StandardScaler()
    chunk['reviewNormalized'] = scaler.fit_transform(chunk[['reviews']]).flatten()
    chunk['boughtNormalized'] = scaler.fit_transform(chunk[['boughtInLastMonth']]).flatten()


    #print(chunk[['reviews', 'reviewNormalized']].head())

    feature_cols = ['price', 'reviews', 'boughtInLastMonth']
    y = (chunk['isBestSeller'])
    X = (chunk[feature_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    df_y_test_total.append(y_test)


    #clf.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)

    data_y_pred.append(y_pred)
    print('chunk proceeded')
    


final_y_test = pd.concat(df_y_test_total)
arr = np.concatenate(data_y_pred).ravel()

cm = confusion_matrix(final_y_test, arr, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=svc.classes_)
disp.plot()
plt.show()