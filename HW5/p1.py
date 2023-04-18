from model import Data_Generation, classifier
import threading
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def LogisticTrials():
    fold = 10
    logitkfolf = KFold(n_splits=fold)
    logitkfolf.get_n_splits(x_train.T)
    results = []
    models = []
    neurons = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for neuron in neurons:
        print(f"Number of neurons: {neuron}")
        acc = 0
        for train_index, test_index in logitkfolf.split(x_train.T):
            x_train_split, x_test_split = x_train.T[train_index], x_train.T[test_index]
            y_train_split, y_test_split = x_train_label[train_index], x_train_label[test_index]

            x_train_split = x_train_split.T
            y_train_split = y_train_split.reshape(-1)

            x_test_split = x_test_split.T
            y_test_split = y_test_split.reshape(-1)
            training = classifier(x_train_split, y_train_split)
            model = training.Logit(1000, 0.001, neuron)
            acc_tmp = training.predict(model, x_test_split, y_test_split, False)
            acc += acc_tmp
            print(acc_tmp)
        print(f"average accuracy of {fold} fold cross validation: {acc/fold}")
        models.append(model)
        results.append(acc/fold)

    return models, results, neurons


if __name__ == "__main__":

    # generate training data
    train = Data_Generation(1000, [0.5, 0.5])
    x_train, x_train_label = train.gen()
    
    # generate test data
    test = Data_Generation(10000, [0.5, 0.5])
    x_test, x_test_label = test.gen()

    '''
    models1, results1, neurons1 = LogisticTrials() 
    models2, results2, neurons2 = LogisticTrials()
    models3, results3, neurons3 = LogisticTrials()

    best = np.argmax(results1)
    best_model = models1[best]
    #test_acc = training.predict(best_model, x_test, x_test_label, True)
    #print(f"Accuracy of test data using optimal parameters: {test_acc}")
    plt.figure(2)
    plt.plot(neurons1, results1)
    plt.plot(neurons2, results2)
    plt.plot(neurons3, results3)
    plt.legend(["Trial 1", "Trial 2", "Trial 3"])
    plt.xlabel("Number of neurons")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1)
    plt.title("Logistic Regression")
    plt.show()
    '''
    training = classifier(x_train, x_train_label)
    model = training.Logit(1000, 0.001, 7)
    test_acc = training.predict(model, x_test, x_test_label, True)
    print(f"Accuracy of test data using optimal parameters: {test_acc}")

    '''    
    classify = classifier(x_train, x_train_label)
    gamma = np.arange(1, 5, 0.5)
    C = np.arange(0.1, 1, 0.05)
    avgall = []
    param = []
    print("Testing Support Vector Machine 10 fold cross validation...")
    for g in gamma:
        for c in C:
            avg = classify.SVM(g, c, None, None, 10)
            print(f"average accuracy for box constraint = {g}, Gwidth = {c}: {avg}")
            sample = [g, c]
            param.append(sample)
            avgall.append(avg)

    maxAcc = np.max(avgall)
    maxParam = param[np.argmax(avgall)]
    param = np.array(param)


    plt.figure(3)
    ax = plt.axes(projection='3d')
    #ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,)
    ax.scatter3D(param[:, 0], param[:, 1], avgall)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('C')
    ax.set_zlabel('Accuracy')
    ax.set_zlim(0.75, 0.85)
    plt.title("Support Vector Machine")
    plt.show()

    acc = classify.SVM(maxParam[0], maxParam[1], x_test, x_test_label,  None)
    print(f"Accuracy of test data using optimal parameters box constraint = {maxParam[0]}, Gwidth = {maxParam[1]}: {acc}")
    '''