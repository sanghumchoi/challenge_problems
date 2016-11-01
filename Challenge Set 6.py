import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation, neighbors, metrics, linear_model, learning_curve, naive_bayes, svm, ensemble, tree
import os

os.chdir("/home/choiboy9106/Desktop/Metis/Challenges/Challenge Set 6")

# Challenge 6.1
vote_df = pd.read_csv('house-votes-84.data', header=None)
vote_df.replace(['y', 'n'], [1, 0], inplace=True)
vote_df = vote_df.apply(lambda x: x.replace('?', sum(x==1)/(x.shape[0]-sum(x=='?'))))
vote_df[16] = vote_df[16].apply(lambda x: x.replace('.', ''))
vote_df[16] = vote_df[16].apply(lambda x: 1 if x=='democrat' else 0)

X = vote_df.iloc[:,:-1]
y = vote_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)

neigh = KNeighborsClassifier(n_neighbors=10)
logit = LogisticRegression()
nb = GaussianNB()
svc = SVC(probability=True)
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

model_names = ['KNN', 'Logistic','Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
models = [neigh, logit, nb, svc, dtc, rfc]
accuracy = []
precision = []
recall = []
f1 = []
for model in models:
    model.fit(X_train, y_train)
    accuracy.append(accuracy_score(model.predict(X_test), y_test))
    precision.append(precision_score(model.predict(X_test), y_test))
    recall.append(recall_score(model.predict(X_test), y_test))
    f1.append(f1_score(model.predict(X_test), y_test))

metrics_df = pd.DataFrame({'model':model_names, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1})
metrics_df.set_index('model', inplace=True)
print(metrics_df)


# Challenge 6.2
def plot_ROC_curve(models, X, y, model_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4444)
    print('AUC Score')
    for name, model in zip(model_names, models):

        model.fit(X_train, y_train)
        pred_probs = model.predict_proba(X_test)

        fpr, tpr, thresholds = roc_curve(y_test,pred_probs[:,1])

        plt.plot(fpr, tpr, label=name);
        plt.legend(loc='best');
        print(name+': %6.4f' % roc_auc_score(y_test,pred_probs[:,1]))
    plt.plot(fpr,fpr, 'r--');
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.show()

plot_ROC_curve(models, X, y, model_names)


# Challenge 6.3
accuracy = []
precision = []
recall = []
f1 = []
for model in models:
    accuracy.append(np.mean(cross_val_score(model, X, y)))
    precision.append(np.mean(cross_val_score(model, X, y, scoring='precision')))
    recall.append(np.mean(cross_val_score(model, X, y, scoring='recall')))
    f1.append(np.mean(cross_val_score(model, X, y, scoring='f1')))

cv_metrics = pd.DataFrame({'model':model_names, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1})
cv_metrics.set_index('model', inplace=True)
cv_metrics


# Challenge 6.4
moviesdf = pd.read_csv('2013_movies.csv')
X = moviesdf[['Gross', 'budget', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama',
       'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']]
mpaa = moviesdf['MPAA']

def pr_metrics(estimator, X, y):
    ratings = np.unique(mpaa.tolist())
    precision = []
    recall = []
    for rating in ratings:
        y = mpaa.apply(lambda x: 1 if x==rating else 0)

        precision.append(np.mean(cross_val_score(estimator, X, y, scoring='precision')))
        recall.append(np.mean(cross_val_score(estimator, X, y, scoring='recall')))

    df = pd.DataFrame({'rating':ratings, 'precision':precision, 'recall':recall})
    df.set_index('rating', inplace=True)
    return df

print(pr_metrics(KNeighborsClassifier(n_neighbors=15), X, y))
print(pr_metrics(LogisticRegression(), X, y))


# Challenge 6.5
habermandf = pd.read_csv('haberman.csv', header=None)
habermandf.columns=['age', 'year', 'nodes', 'status']
habermandf['status'] = habermandf['status'].apply(lambda x: 0 if x==2 else x)

X = habermandf.drop('status', axis=1)
y = habermandf['status']

plot_ROC_curve([logit], X, y,['Logistic'])
