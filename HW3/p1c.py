from model import twoClassProblem
from model import classifier
import numpy as np


if __name__ == "__main__":
    meam_01 = [5,0]

    cov_01 = [[4,0],
              [0,2]]
    meam_02 = [0,4]
    cov_02 = [[1,0],
              [0,3]]
    
    mean0 = [meam_01, meam_02]
    cov0 = [cov_01, cov_02]

    mean1 = [3,2]
    cov1 = [[2,0],
            [0,3]]
    
    mean = [mean0, mean1]
    cov = [cov0, cov1]
    priors = [0.6, 0.4]
    iter = 1000
    # test data
    test = twoClassProblem(priors, mean, cov, N = 200000)
    test_x, test_label_x = test.generateTestData()

    # train with 10000 samples
    print('train with 10000 samples')
    train = twoClassProblem(priors, mean, cov, N = 10000)
    train_x, train_label_x = train.generateTestData()
    train.plotTrueClass()
    classify = classifier(train_x, train_label_x, priors, mean, cov)
    print('linear layer')
    y_pred = classify.Logit(iter, 0.001, False, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)

    print('nonlinear layer')
    y_pred = classify.Logit(iter, 0.001, True, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)
    print("\n")

    # train with 1000 samples
    print('train with 1000 samples')
    train = twoClassProblem(priors, mean, cov, N = 1000)
    train_x, train_label_x = train.generateTestData()
    train.plotTrueClass()
    classify = classifier(train_x, train_label_x, priors, mean, cov)
    print('linear layer')
    y_pred = classify.Logit(iter, 0.001, False, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)

    print('nonlinear layer')
    y_pred = classify.Logit(iter, 0.001, True, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)
    print("\n")

    # train with 100 samples
    print('train with 100 samples')
    train = twoClassProblem(priors, mean, cov, N = 100)
    train_x, train_label_x = train.generateTestData()
    train.plotTrueClass()
    classify = classifier(train_x, train_label_x, priors, mean, cov)
    print('linear layer')
    y_pred = classify.Logit(iter, 0.001, False, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)
    print('nonlinear layer')
    y_pred = classify.Logit(iter, 0.001, True, test_x)
    y_pred = y_pred.squeeze()
    incorrect = np.where(y_pred != test_label_x)[0]
    p_error = len(incorrect)/len(y_pred)
    print('p_error: ', p_error)
    print("\n")



