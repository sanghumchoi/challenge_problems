import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Challenge 4.1
x = np.random.randint(1,5000, size = (50000,1))
y = 2000 + 2.5*np.log(x) + np.random.randint(1,100)
model = linear_model.LinearRegression()
model.fit(x, y)
print("Regular Linear Regression")
print("Score: ", model.score(x, y))
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

x1 = x**2
model1 = linear_model.LinearRegression()
model1.fit(x1, y)
print("Quadrtic Simulation")
print("Score: ", model1.score(x1, y))
print("Intercept: ", model1.intercept_)
print("Coefficient: ", model1.coef_)

x2 = np.log(x)
model2 = linear_model.LinearRegression()
model2.fit(x2, y)
print("Logarithmic Simulation")
print("Score: ", model2.score(x2, y))
print("Intercept: ", model2.intercept_)
print("Coefficient: ", model2.coef_)


# Challenge 4.2
a = np.random.randint(1,5000, size = (50000,1))
b = 2000 + 2.5*a + 1.25*(a**2) + np.random.randint(1,100)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(b, a, test_size = 0.25)

model3 = linear_model.LinearRegression()
model3.fit(x_train, y_train)
print("Score: ", model3.score(x_test, y_test))
print("Intercept: ", model3.intercept_)
print("Coefficient: ", model3.coef_)

ypredict = model3.predict(x_train)
print('Train MSE: ', mean_squared_error(y_train, ypredict))
ypredict1 = model3.predict(x_test)
print('Test MSE: ', mean_squared_error(y_test, ypredict1))


# Challenge 4.3
msetrain = []
msetest = []
rsquared = []
aic = []
for i in range(8):
    xpoly = x**i
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(xpoly, y, test_size = 0.25)
    x_train = sm.add_constant(x_train)
    ols_model = sm.OLS(y_train, x_train).fit()
    msetrain.append(ols_model.mse_resid)

    x_test = sm.add_constant(x_test)
    ols_model = sm.OLS(y_test, x_test).fit()
    msetest.append(ols_model.mse_resid)
    rsquared.append(ols_model.rsquared)
    aic.append(ols_model.aic)

MSE_df = pd.DataFrame({'msetrain': msetrain, 'msetest': msetest, 'Degree': range(8)})
MSE_df.set_index('Degree', inplace = True)
plt.show(MSE_df.plot())

_, ax1 = plt.subplots()
plt.show(ax1.plot(range(8), rsquared, "b-"))
ax1.set_xlabel("Degree")

ax2 = ax1.twinx()
plt.show(ax2.plot(range(8), aic, "g-"))
ax2.set_ylabel("AIC")


# Challenge 4.4
x_train1, x_test1, y_train1, y_test1 = cross_validation.train_test_split(x, y, test_size = 0.25)
interval = np.arange(5, x.shape[0], 5)
mse_train1 = []
mse_test1 = []
for i in interval:
    X_tr = sm.add_constant(x_train1[:i])
    ols_model = sm.OLS(y_train1[:i], X_tr).fit()
    mse_train1.append(ols_model.mse_resid)

    X_tt = sm.add_constant(x_test1[:i])
    ols_model = sm.OLS(y_test1[:i], X_tt).fit()
    mse_test1.append(ols_model.mse_resid)

MSE_df = pd.DataFrame({'mse_train1': mse_train1, 'mse_test1': mse_test1, 'interval': interval})
MSE_df.set_index('i', inplace = True)
plt.show(MSE_df.plot())
