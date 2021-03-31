# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix

data = pd.read_csv(r'C:\Users\Dinesh Kaarthik\Desktop\SVM\Cellphone.csv')
data[:3]

X = data.drop('price_range', axis = 1)
y = data['price_range']

y.unique()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)

svm = SVC()
svm.fit(x_train, y_train)
score_before_scaling = svm.score(x_test, y_test)

#feature Scaling
X = (X - np.min(X)) / (np.max(X) - np.min(X))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)

svm = SVC()
svm.fit(x_train, y_train)
score_before_scaling = svm.score(x_test, y_test)


#Feature selection

train_accuracy = []
k = np.arange(1, 21)

for i in k:
    select = SelectKBest(f_classif, k=i)
    x_train_new = select.fit_transform(x_train, y_train)
    svm.fit(x_train_new, y_train)
    train_accuracy.append(svm.score(x_train_new, y_train))
    

plt.plot(k, train_accuracy, color = 'red', label = 'Train')
plt.xlabel('k values')
plt.ylabel('Train accuracy')
plt.legend()
plt.show()


select_top = SelectKBest(f_classif, k=5)
x_train_new = select_top.fit_transform(x_train, y_train)
x_test_new = select_top.fit_transform(x_test, y_test)
print('Top features for train:', x_train.columns.values[select_top.get_support()])
print('Top features for test:', x_test.columns.values[select_top.get_support()])

c = [1.0, 0.25, 0.5, 0.75]
kernels = ['linear', 'rbf']
gammas = ['auto', 0.01, 0.001, 1]


#Grid search for SVM
svm = SVC()

grid_svm = GridSearchCV(estimator = svm, param_grid = dict(kernel = kernels,
                                                           C = c,
                                                           gamma = gammas), cv = 5)
grid_svm.fit(x_train_new, y_train)
print('The best hyper parameters are:', grid_svm.best_estimator_)

svc_model = SVC(C=1.0, gamma='auto', kernel='linear')
svc_model.fit(x_train_new, y_train)

print('The train accuracy:', svc_model.score(x_train_new, y_train))
print('The test accuracy:', svc_model.score(x_test_new, y_test))


y_pred = svc_model.predict(x_test_new)
accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred )

cm = ConfusionMatrix(svc_model, classes = [0, 1, 2, 3])
cm.fit(x_train_new, y_train)
cm.score(x_test_new, y_test)
cm.poof()

















