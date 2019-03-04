##
# Do not modify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

df = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv',
    sep=',')
df.head()


##
# Do not modify
df = df.drop(['cbwd'], axis=1)  # drop non-scalar feature
df = df.dropna(axis=0, how='any')  # drop samples who has nan feature
df.head()
##
# Do not modify
idx = np.logical_or(
    np.logical_and(df['year'].values == 2014, df['month'].values < 3),
    np.logical_and(df['year'].values == 2013, df['month'].values == 12))
X = df.loc[idx].drop('pm2.5', axis=1)
y = df.loc[idx]['pm2.5'].values
X.head()

##
# To simplify our codes, predefine a function to visualize to regression line and data scatter plot.
def lin_regplot(X, y, model):
  plt.scatter(X, y, c='blue')
  plt.plot(X, model.predict(X), color='red', linewidth=2)
  return

##
X.columns = [
'No',  'year',  'month'  ,'day'  ,'hour',  'DEWP',  'TEMP',  'PRES',   'Iws',  'Is', 'Ir'
]

X.head()

##
import matplotlib.pyplot as plt

x_vars = [
'No',  'year',  'month'  ,'day'  ,'hour',  'DEWP',  'TEMP',  'PRES',   'Iws',  'Is', 'Ir'

]

_, subplot_arr = plt.subplots(3, 5, figsize=(20, 12))
for idx, x_var in enumerate(x_vars):
  x_idx = idx // 5
  y_idx = idx % 5
  subplot_arr[x_idx, y_idx].scatter(df[x_var], df['pm2.5'])
  subplot_arr[x_idx, y_idx].set_xlabel(x_var)

plt.show()



##
from sklearn.linear_model import LinearRegression
import numpy as np

X_lws = X['Iws'].values[:, np.newaxis]

slr = LinearRegression()
# fit
slr.fit(X_lws, y)

y_pred = slr.predict(X_lws)

print('Slope (w_1): %.2f' % slr.coef_[0])
print('Intercept/bias (w_0): %.2f' % slr.intercept_)

##
lin_regplot(X_lws, y, slr)
plt.xlabel('Lws')
plt.ylabel('PM2.5')
plt.tight_layout()
plt.show()


##
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X_lws)
quad_regr = LinearRegression()
quad_regr.fit(X_quad, y)
quadratic_r2 = r2_score(y, quad_regr.predict(X_quad))


# plot results

X_range = np.arange(X_lws.min(), X_lws.max(), 1)[:, np.newaxis]
y_quad_pred = quad_regr.predict(quadratic.fit_transform(X_range))

plt.scatter(X_lws, y, label='Training points', color='lightgray')
plt.plot(
    X_range,
    y_quad_pred,
    label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
    color='red',
    lw=2,
    linestyle='-')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



