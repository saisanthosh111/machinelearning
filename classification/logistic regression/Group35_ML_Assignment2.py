import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# calculating the accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if predicted[i]==1.0:
            if actual[i]=="Iris-versicolor":
                correct += 1
        elif predicted[i]==0.0:
            if actual[i]=="Iris-virginica":
                correct+=1
    return correct / float(len(actual)) * 100.0

# because of two classes we are representing 0.0 with Iris-versicolor and 1.0 with Iris-virginica
def change(y):
    if y=="Iris-virginica":
        return 0.0
    else:
        return 1.0


# evaluating the row with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + np.exp(-yhat))

# returning the required coefficients using epoch
def epoch_stochastic(train_data, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train_data.iloc[0,:]))]
    for epoch in range(n_epoch):
        for row in train_data.values:
            yhat = predict(row, coef)
            x = change(row[-1])
            error = x - yhat
            coef[0] = coef[0] + l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * row[i]
    return coef

# returning the required coefficients using euclidian norm
def euclidian_stochastic(train_data, l_rate):
    coef = [0.0 for i in range(len(train_data.iloc[0,:]))]
    count = 0
    while(True):
        for row in train_data.values:
            yhat = predict(row, coef)
            x = change(row[-1])
            error = x - yhat
            z = error * yhat * (1.0 - yhat)
            coef[0] = coef[0] + l_rate * z
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * z*row[i]
        if abs(z)<0.01:
            return coef
    return coef

# evaluation of test set with epoch
def epoch_logistic_regression(train_data,test_data,l_rate,n_epoch):
    predictions = list()
    coef = epoch_stochastic(train_data, l_rate, n_epoch)
    for row in test_data.values:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    actual = [row for row in test_data.iloc[:,-1]]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy

# evaluation of test set with euclidian norm
def euclidian_logistic_regression(train_data,test_data,l_rate):
    predictions = list()
    coef = euclidian_stochastic(train_data, l_rate)
    for row in test_data.values:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    actual = [row for row in test_data.iloc[:,-1]]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy


if __name__ == "__main__":

#     loading the datasets
    train_data = pd.read_excel("D:/thopu/train_2.xlsx")
    test_data = pd.read_csv("D:/thopu/test_2.csv")

    # analysing the datasets
    x = [change(i) for i in train_data.iloc[:,-1]]
    y = ["Iris-versicolor","iris-virginica"]
    plt.scatter(train_data.iloc[:, 2],train_data.iloc[:, 3], c=x,cmap=plt.cm.Paired, s=100)

    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()

    plt.hist(train_data.iloc[:,3])
    plt.xlabel("petal width")
    plt.show()

    plt.hist(train_data.iloc[:,2])
    plt.xlabel("petal length")
    plt.show()

    plt.hist(train_data.iloc[:,1])
    plt.xlabel("sepal width")
    plt.show()

    plt.hist(train_data.iloc[:,0])
    plt.xlabel("sepal length")
    plt.show()

#     Training and evaluating the iris data set with the input of epochs
    score = epoch_logistic_regression(train_data,test_data,0.02,80)

#     Training and evaluating the iris data set with Euclidian norm
    score1 = euclidian_logistic_regression(train_data,test_data,0.02)

    print("The accuracy of our model stopping with Euclidian norm is {}".format(score))

    print("The accuracy of our model using epochs is {}".format(score1))
