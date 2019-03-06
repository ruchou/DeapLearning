##
import pandas as pd
import numpy as np

#load the data
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                   'arrhythmia/arrhythmia.data', header=None, sep=',', engine='python')

display(data.head(3))

print('%d rows and %d columns' % (data.shape[0],data.shape[1]))
np.unique(data[len(data.columns)-1])

data['arrhythmia'] = data[len(data.columns)-1].map(lambda x: 0 if x==1 else 1)
data = data.drop(len(data.columns)-2, axis=1)
data.groupby(['arrhythmia']).size()
data = data._get_numeric_data()
print('%d rows and %d columns' % (data.shape[0],data.shape[1]))
data.head(3)

X = data.iloc[:, :-1]  # The first to second-last columns are the features
y = data.iloc[:, -1]   # The last column is the ground-truth label
print(np.unique(y))
print(X.shape)


# splitting the dataset to training and validation datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20181004)

# Standardizing the training and test datasets
# Note that we are scaling based on the information from the training data
# Then we apply the scaling that is done from training data to the test data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

##
class LogisticRegression(object):

    def __init__(self, eta=0.05, n_epoch=100, random_state=1):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_epoch):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = self.loss(output, y)
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def loss(self, output, y):
        """Calculate loss"""
        # TODO
        return (-y * np.log(output) - (1 - y) * np.log(1 - output)).mean()
        pass

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        # TODO
        return  1.0 / (1.0+np.exp(-z))
        pass

    def predict(self, X):
        """Return class label after unit step"""
        # TODO
        prob=self.activation(self.net_input(X))

        return np.where(prob < 0.5, 0, 1)

        pass
##
from sklearn.metrics import *
import matplotlib.pyplot as plt

epoch = 300
lr = 1e-3

reg = LogisticRegression(eta=lr, n_epoch=epoch)
reg.fit(X_train_std, y_train)
y_test_pred = reg.predict(X_test_std)
acc = accuracy_score(y_test, y_test_pred)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(epoch), reg.cost_, label=f'Loss(Acc={acc:.3f})')
ax.legend()
plt.show()

##
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_test_pred), cmap=plt.cm.Blues, annot=True, ax=ax)
ax.set_xlabel('Pred Label')
ax.set_ylabel('True Label')
plt.show()
##

print(classification_report(y_test, y_test_pred))
