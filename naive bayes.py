import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('buy_computer.csv')
print(df.head(14))
df_enc = df.apply(LabelEncoder().fit_transform)
y = df_enc['buys_computer'].values
print(y)
X = df_enc[['age', 'income', 'student', 'credit_rating']].values
print(X)


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

iris = datasets.load_iris()
#X, y = iris.data, iris.target

print(X.shape)
print(X)
print(y)
#%%
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
#%% 
nb = NaiveBayes()
nb.fit(X, y)
preds = nb.predict(X)

acc = np.sum(preds == y)/len(y)

#%%
print(acc, preds)

preds = nb_classifier.predict(X)
print("Confusion matrix: ")
print(cm(y, preds))
print("Metrics: ")
print(cr(y, preds))
print("Accuracy: ", np.sum(preds == y)/len(y))

#age <=30,Income = medium,Student = yes,Credit_rating = fair = [1, 2, 1, 1]
preds = nb_classifier.predict([[1, 2, 1, 1]])
for pred in preds:
    if(pred == 1):
        print("yes")
        
        
        
        
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb_classifier = NaiveBayes()
nb_classifier.fit(X_train, y_train)

preds = nb_classifier.predict(X_train)
print("Confusion matrix: ")
print(cm(y_train, preds))
print("Metrics: ")
print(cr(y_train, preds))
print("Accuracy: ", np.sum(preds == y_train)/len(y_train))
