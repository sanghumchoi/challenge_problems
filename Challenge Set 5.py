import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, neighbors, metrics, linear_model, learning_curve, naive_bayes, svm, ensemble, tree
from sklearn.svm import SVC
import os

os.chdir("/home/choiboy9106/Desktop/Metis/Challenges/Challenge Set 5")

# Challenge 5.1
votingdf = pd.read_csv("house-votes-84.data", header = None)
votingdf.columns = ["party", "vote 1", "vote 2", "vote 3", "vote 4","vote 5", "vote 6", "vote 7", "vote 8",
                    "vote 9", "vote 10", "vote 11", "vote 12", "vote 13", "vote 14", "vote 15", "vote 16"]
votingdf = votingdf.replace(["y", "n", "?"], [1, 0, np.NaN])
print(votingdf.head())
votingdf = votingdf.fillna(votingdf.mean())
print(votingdf)

# Challenge 5.2
x = votingdf.iloc[:,1:]
y = votingdf.iloc[:,:1]
# print(x)
print(y)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.3, random_state = 4444)


# Challenge 5.3
def knc(i):
    neigh = neighbors.KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x_train, y_train)
    predict = neigh.predict(x_test)
    print(predict)
    print(metrics.accuracy_score(y_test, predict))
knc(9)


# Challenge 5.4
def logregression():
    log_regression = linear_model.LogisticRegression()
    log_regression.fit(x_train, y_train)
    predict_log = log_regression.predict(x_test)
    print(predict_log)
    print(metrics.accuracy_score(y_test, predict_log))
logregression()


# Challenge 5.5
partycount = y["party"].value_counts()
demcount = partycount["democrat"]
repcount = partycount["republican"]
# print(demcount)
# print(repcount)

objects = ["Democrat", "Republican"]
govcount = [demcount, repcount]
y_pos = np.arange(len(objects))
plt.xticks(y_pos, objects)
plt.ylabel("Number of Government Officials")
plt.xlabel("Party Designation")
plt.title("Count of Democrats and Republicans")
plt.show(plt.bar(y_pos, govcount, align = "center", alpha = 0.5))

# possibly fix this
def democrat(x):
    return ["democrat" for i in x]
# print(metrics.accuracy_score(y_test, democrat(435)))

def republican(x):
    return ["republican" for i in x]
# print(metrics.accuracy_score(y_test, republican(435)))


# Challenge 5.6



# Challenge 5.7
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel = 'logarithmic'), x, y, train_sizes = [50, 80, 110], cv = 5)
print(train_sizes)
print(train_scores)
print(SVC)


# Challenge 5.8
def gaussian():
    gaussianreg = naive_bayes.GaussianNB()
    gaussianreg.fit(x_train, y_train)
    predictgauss = gaussianreg.predict(x_test)
    print(predictgauss)
    print(metrics.accuracy_score(y_test, predictgauss))
gaussian()

def supportvm():
    svmreg = svm.SVC()
    svmreg.fit(x_train, y_train)
    predictsvm = svmreg.predict(x_test)
    print(predictsvm)
    print(metrics.accuracy_score(y_test, predictsvm))
supportvm()

def decisiontree():
    treereg = tree.DecisionTreeClassifier()
    treereg.fit(x_train, y_train)
    predicttree = treereg.predict(x_test)
    print(predicttree)
    print(metrics.accuracy_score(y_test, predicttree))
decisiontree()

def randomforest():
    forestreg = ensemble.RandomForestClassifier()
    forestreg.fit(x_train, y_train)
    predictforest = forestreg.predict(x_test)
    print(predictforest)
    print(metrics.accuracy_score(y_test, predictforest))
randomforest()


# Challenge 5.9



# Challenge 5.10
votingdf2 = pd.read_csv("house-votes-84.data", header = None)
votingdf2.columns = ["party", "vote 1", "vote 2", "vote 3", "vote 4","vote 5", "vote 6", "vote 7", "vote 8",
                    "vote 9", "vote 10", "vote 11", "vote 12", "vote 13", "vote 14", "vote 15", "vote 16"]
votingdf2 = votingdf2.replace(["y", "n", "?"], [1, 0, np.NaN])
print(votingdf2.head())
votingdf2 = votingdf2.fillna(votingdf2.mode().iloc[0]) #figure out how to populate NaN with mode of series
print(votingdf2.head())
a = votingdf2.iloc[:,2:]
b = votingdf2.iloc[:,1:2]
# print(a)
# print(b)
a_train, a_test, b_train, b_test = cross_validation.train_test_split(a, b, test_size = 0.3, random_state = 4444)

def knc1(i):
    neigh = neighbors.KNeighborsClassifier(n_neighbors = i)
    neigh.fit(a_train, b_train)
    predict = neigh.predict(a_test)
    print(predict)
    print(metrics.accuracy_score(b_test, predict))
knc1(9)

def logregression1():
    log_regression = linear_model.LogisticRegression()
    log_regression.fit(a_train, b_train)
    predict_log = log_regression.predict(a_test)
    print(predict_log)
    print(metrics.accuracy_score(b_test, predict_log))
logregression1()

def gaussian1():
    gaussianreg = naive_bayes.GaussianNB()
    gaussianreg.fit(a_train, b_train)
    predictgauss = gaussianreg.predict(a_test)
    print(predictgauss)
    print(metrics.accuracy_score(b_test, predictgauss))
gaussian1()

def supportvm1():
    svmreg = svm.SVC()
    svmreg.fit(a_train, b_train)
    predictsvm = svmreg.predict(a_test)
    print(predictsvm)
    print(metrics.accuracy_score(b_test, predictsvm))
supportvm1()

def decisiontree1():
    treereg = tree.DecisionTreeClassifier()
    treereg.fit(a_train, b_train)
    predicttree = treereg.predict(a_test)
    print(predicttree)
    print(metrics.accuracy_score(b_test, predicttree))
decisiontree1()

def randomforest1():
    forestreg = ensemble.RandomForestClassifier()
    forestreg.fit(a_train, b_train)
    predictforest = forestreg.predict(a_test)
    print(predictforest)
    print(metrics.accuracy_score(b_test, predictforest))
randomforest1()


# Challenge 5.11
os.chdir("/home/choiboy9106/ds/metisgh/nyc16_ds9/challenges/challenges_data")
moviesdf = pd.read_csv("2013_movies.csv")
print(moviesdf.head())
moviesdf = moviesdf[pd.notnull(moviesdf["Budget"])]
moviesdf = moviesdf[pd.notnull(moviesdf["Title"])]
moviesdf = moviesdf[pd.notnull(moviesdf["DomesticTotalGross"])]
moviesdf = moviesdf[pd.notnull(moviesdf["Rating"])]
moviesdf = moviesdf[pd.notnull(moviesdf["Runtime"])]
moviesdf = moviesdf[pd.notnull(moviesdf["ReleaseDate"])]
moviesdf = moviesdf[pd.notnull(moviesdf["Director"])]

plt.show(moviesdf.hist("Rating", weights = moviesdf["Rating"]))

def moviesknc(a, b, i):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(a, b, test_size = 0.3, random_state = 4444)
    neigh = neighbors.KNeighborsClassifier(n_neighbors = i)
    neigh.fit(x_train, y_train)
    predict = neigh.predict(x_test)
    print(predict)
    print(metrics.accuracy_score(y_test, predict))
moviesknc(moviesdf.iloc[:,1:3], moviesdf.iloc[:,4:5],10)

def movieslogregression(c, d):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(c, d, test_size = 0.3, random_state = 4444)
    log_regression = linear_model.LogisticRegression()
    log_regression.fit(x_train, y_train)
    predict_log = log_regression.predict(x_test)
    print(predict_log)
    print(metrics.accuracy_score(y_test, predict_log))
    print(log_regression.coef_)
movieslogregression(moviesdf.iloc[:, 1:3], moviesdf.iloc[:, 4:5])


# Challenge 5.12 - finish
cancerdf = pd.read_csv("haberman.data", header = None)
cancerdf.columns = ["age", "year", "nodes", "survival"]
print(cancerdf.head())

# Average and standard deviation of the age of all of the patients
print(cancerdf["age"].mean())
print(cancerdf["age"].std())

# Average and standard deviation of the age of those patients that survived 5 or more years after surgery
# Average and standard deviation of the age of those patients who survived fewer than 5 years after surgery
print(cancerdf["age"].groupby(cancerdf["survival"]).mean())
print(cancerdf["age"].groupby(cancerdf["survival"]).std())

# Plot a histogram of the ages side by side with a histogram of the number of axillary nodes
age = cancerdf["age"]
nodes = cancerdf["nodes"]
agenodes = pd.concat([age, nodes], axis = 1)
plt.show(agenodes.plot.bar())

# What is the earliest year of surgery in this dataset?
print(min(cancerdf["year"]))

# What is the most recent year of surgery?
print(max(cancerdf["year"]))

# Use logistic regression to predict survival after 5 years. How well does your model do?
# What are the coefficients of logistic regression? Which features affect the outcome how?
def cancerlogregression(i, j):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(i, j, test_size = 0.3, random_state = 4444)
    log_regression = linear_model.LogisticRegression()
    log_regression.fit(x_train, y_train)
    predict_log = log_regression.predict(x_test)
    print(predict_log)
    print(metrics.accuracy_score(y_test, predict_log))
    print(log_regression.coef_)
cancerlogregression(cancerdf.iloc[:,:-1], cancerdf.iloc[:,-1:])
# The older a person gets, the more likely that person is going to not survive the 5 years.
# The earlier a person got surgery, the more likely that person is going to survive the 5 years.
# The more nodes the person has that is affected, the more likely that person is going to not survive the 5 years.
# The nodes has the biggest magnitude effect on survival.

# Draw the learning curve for logistic regression in this case
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel = 'logarithmic'), cancerdf.iloc[:,:-1], cancerdf.iloc[:,-1:], train_sizes = [50, 80, 110], cv = 5)
print(train_sizes)
print(train_scores)
print(SVC)
