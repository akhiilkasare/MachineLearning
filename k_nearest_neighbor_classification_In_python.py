#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# loading datasets

iris = datasets.load_iris()

# printing description 

print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

# training the classifies

clf = KNeighborsClassifier()
clf.fit(features, labels)
pred = clf.predict([[18,1,1,1]])
print(pred)

