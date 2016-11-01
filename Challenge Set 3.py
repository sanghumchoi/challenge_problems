import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation, datasets, grid_search
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import os

os.chdir("/home/choiboy9106/ds/metisgh/nyc16_ds9/challenges/challenges_data")

df = pd.read_csv("2013_movies.csv")
df = df[pd.notnull(df["Budget"])]
df = df[pd.notnull(df["Title"])]
df = df[pd.notnull(df["DomesticTotalGross"])]
df = df[pd.notnull(df["Rating"])]
df = df[pd.notnull(df["Runtime"])]
df = df[pd.notnull(df["ReleaseDate"])]
df = df[pd.notnull(df["Director"])]
model = LinearRegression(fit_intercept = True)
# print(df)
# print(df.describe())

# Challenge 3.1
df["Constants"] = [1 for i in range(len(df))]
x = df.ix[:,["Constants"]]
y = df.ix[:,"DomesticTotalGross"]
model.fit(x,y)
print(model.score(x,y))
print(model.coef_)
print(model.intercept_)

ytheoretical = model.predict(x)
plt.show(plt.scatter(ytheoretical, y))

residual = y - ytheoretical
plt.show(plt.hist(residual))
"""
The model predicts the mean of the dependent variable. The residuals are distributed
with right skew.
"""

# Challenge 3.2
x1 = df.ix[:,["Budget"]]
y1 = df.ix[:,"DomesticTotalGross"]
model.fit(x1,y1)
print(model.score(x1,y1))
print(model.coef_)
print(model.intercept_)

ytheoretical1 = model.predict(x1)
plt.show(plt.scatter(ytheoretical1, y1))

residual1 = y1 - ytheoretical1
plt.show(plt.hist(residual1))

# Challenge 3.3
df["PG"] = pd.get_dummies(df["Rating"])["PG"]
x2 = df.ix[:,["PG"]]
y2 = df.ix[:,"DomesticTotalGross"]
model.fit(x2,y2)
print(model.score(x2,y2))
print(model.coef_)
print(model.intercept_)

ytheoretical2 = model.predict(x2)
plt.show(plt.scatter(ytheoretical2, y2))

residual2 = y2 - ytheoretical2
plt.show(plt.hist(residual2))

# Challenge 3.4 & 3.5
df["PG"] = pd.get_dummies(df["Rating"])["PG"]
df["PG-13"] = pd.get_dummies(df["Rating"])["PG-13"]
df["R"] = pd.get_dummies(df["Rating"])["R"]
x3 = df.ix[:,["G", "PG", "PG-13", "R", "Runtime", "Budget"]]
y3 = df.ix[:,"DomesticTotalGross"]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x3, y3, test_size = 0.25)

lasso = Lasso(normalize=True)

parameters = {'normalize':(True,False), 'alpha':np.logspace(-10,1,100)}
grid_searcher = grid_search.GridSearchCV(lasso, parameters)
grid_searcher.fit(x_train, y_train)
print(grid_searcher.best_params_)

models = {}
models['ridge'] = linear_model.Ridge()
models['lasso'] = linear_model.Lasso(alpha = .2)
models['elasticnet'] = linear_model.ElasticNet()

for name, model in models.items():
    model.fit(x_train,y_train)
    print('Model: ' + name)
    print("Score: " + str(model.score(X_test, y_test)))
    sorted_features = sorted(zip(data_body.columns, model.coef_), key = lambda tup: abs(tup[1]), reverse = True)
    for feature in sorted_features:
        print(feature)
    print("")
