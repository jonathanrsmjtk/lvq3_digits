from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import math

#Train using LVQ 3
def lvq3_train(X, y, a, b, max_ep, min_a, e):
    c, train_idx = np.unique(y, True)
    r = c
    W = X[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(zip(X, y)) if i not in train_idx])
    X = train[:, 0]
    y = train[:, 1]
    ep = 0

    while ep < max_ep and a > min_a:
        for i, x in enumerate(X):
            d = [math.sqrt(sum((w - x) ** 2)) for w in W]
            min_1 = np.argmin(d)

            min_2 = 0
            dc = float(np.amin(d))
            dr = 0
            min_2 = d.index(sorted(d)[1])
            dr = float(d[min_2])
            if c[min_1] == y[i] and c[min_1] != r[min_2]:
                W[min_1] = W[min_1] + a * (x - W[min_1])

            elif c[min_1] != r[min_2] and y[i] == r[min_2]:
                if dc != 0 and dr != 0:

                    if min((dc/dr),(dr/dc)) > (1-e) / (1+e):
                        W[min_1] = W[min_1] - a * (x - W[min_1])
                        W[min_2] = W[min_2] + a * (x - W[min_2])
            elif c[min_1] == r[min_2] and y[i] == r[min_2]:
                W[min_1] = W[min_1] + e * a * (x - W[min_1])
                W[min_2] = W[min_2] + e * a * (x- W[min_2])
        a = a * b
        ep += 1
    return W, c

#Test Using LVQ 3
def lvq3_test(x, W):
    
    W, c = W
    d = [math.sqrt(sum((w - x) ** 2)) for w in W]

    return c[np.argmin(d)]

#Evaluation
def print_metrics(labels, preds):
    print("Precision Score: {}".format(precision_score(labels,
           preds, average = 'weighted')))
    print("Recall Score: {}".format(recall_score(labels, preds,
           average = 'weighted')))
    print("Accuracy Score: {}".format(accuracy_score(labels,
           preds)))
    print("F1 Score: {}".format(f1_score(labels, preds, average =
           'weighted')))

#Load Data
digits = datasets.load_digits()

#Load the digits dataset
fig = plt.figure
plt.imshow(digits.images[-2], cmap='gray_r')
print(plt.show())

#Split to X and Y
X = digits.data
Y = digits.target

#Split to train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)

#Train
W = lvq3_train(x_train, y_train, 0.2, 0.5, 100, 0.001, 0.3)

#Test and predict
predicted = []
for i in x_test:
    predicted.append(lvq3_test(i, W))
print_metrics(y_test, predicted)

#Test the data and compare to real target
print("Target by classifying: ", lvq3_test(x_test[2], W))
print("Real target: ", y_test[2])