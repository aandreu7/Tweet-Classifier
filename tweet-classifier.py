import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from Naive_Bayes_Classifier import Naive_Bayes


data = pd.read_csv("./FinalStemmedSentimentAnalysisDataset.csv", sep=";")
data = data.drop("tweetId", axis=1)

data = data.dropna().reset_index(drop=True)

X = data["tweetText"]
y = data["sentimentLabel"]

model = Naive_Bayes(alpha=1)

results = []

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=777)
accuracies = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracies.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracies))