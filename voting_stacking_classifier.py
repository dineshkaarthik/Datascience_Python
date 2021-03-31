# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv(r'C:\Users\Dinesh Kaarthik\Desktop\VS\diabetes_data.csv')

X = df.drop(columns = ['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


#create a knn model
knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_grid = GridSearchCV(knn, params_knn, cv=5)
knn_grid.fit(X_train, y_train)

knn_final = knn_grid.best_estimator_
print(knn_grid.best_params_)

#create a randomforest classifier
rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
rf_grid = GridSearchCV(rf, params_rf, cv=5)

rf_grid.fit(X_train, y_train)


rf_best = rf_grid.best_estimator_
print(rf_grid.best_params_)

#create a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

#Test accuracy
print('knn: {}'.format(knn_final.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))

#create a dictionary of our models
estimators=[('knn', knn_final), ('rf', rf_best), ('log_reg', log_reg)]

#create voting classifier
vc = VotingClassifier(estimators)
vc.fit(X_train, y_train)
vc.score(X_test, y_test)


#Simple Stacking CV classifier
clf1 = KNeighborsClassifier(n_neighbors=10)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr,
                            random_state=42)
sclf.fit(X_train, y_train)
sclf.score(X_test, y_test)

#Stacking classifier using probabilities as Meta-Features
clf1 = KNeighborsClassifier(n_neighbors=10)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr, use_probas=True,
                            random_state=42)
sclf.fit(X_train, y_train)
sclf.score(X_test, y_test)


