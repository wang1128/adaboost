__author__ = 'penghao'
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import *

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

infile = np.load('train.npy')
testfile = np.load('test.npy')
x=infile[:,:200]
y= infile[:,200]
iris = load_iris()

print(y)
#clf = AdaBoostClassifier(n_estimators=400)


#scores = cross_val_score(clf, x, y)
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),n_estimators=500, learning_rate=1)
scores = cross_val_score(clf, x, y)

#clf2 = AdaBoostClassifier(SVC(probability=True,kernel='rbf'),n_estimators=100, learning_rate=1.0, algorithm='SAMME')
#clf2.fit(x, y)
#scores2 = cross_val_score(clf2, x, y)
#print(scores)
print(scores)