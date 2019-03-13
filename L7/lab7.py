##

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler


import sys
sys.path.append("../L4")

from Tools import plot_decision_regions


##
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'mushroom/agaricus-lepiota.data', header=None, engine='python')
column_name = ['classes','cap-shape', 'cap-surface','cap-color','bruises?','odor',
               'gill-attachment','gill-spacing','gill-size','gill-color',
               'stalk-shape','stalk-root','stalk-surface-above-ring',
               'stalk-surface-below-ring','stalk-color-above-ring',
               'stalk-color-below-ring','veil-type','veil-color','ring-number',
               'ring-type','spore-print-color','population','habitat']
df.columns = column_name
df.head()


##



X = df.drop('classes', axis=1)
y = df['classes'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

##
transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values='?', strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(
    transformers=[('cat', transformer, df.columns[1:])])



##
clf_knn = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', KNeighborsClassifier())
]).fit(X_train, y_train)

acc = clf_knn.score(X_test, y_test)
print(acc)

##

clf_svc = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', SVC(gamma='scale'))
])
acc = clf_svc.fit(X_train, y_train).score(X_test, y_test)
print(acc)
##

from sklearn.model_selection import GridSearchCV


param_gamma = [0.0001, 0.001, 0.01, 0.1, 1.0]
param_C = [0.1, 1.0, 10.0, 100.0]
# here you can set parameter for different steps
# by adding two underlines (__) between step name and parameter name
param_grid = [{'clf__C': param_C,
               'clf__kernel': ['linear']},
              {'clf__C': param_C,
               'clf__gamma': param_gamma,
               'clf__kernel': ['rbf']}]

##


# set pipe_svm as the estimator
gs = GridSearchCV(estimator=clf_svc,
                  param_grid=param_grid,
                  scoring='accuracy')

gs = gs.fit(X_train, y_train)
print('[SVC: grid search]')
print('Validation accuracy: %.3f' % gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
