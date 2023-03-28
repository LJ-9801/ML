from model import twoClassProblem
from model import classifier

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

    # train with 10000 samples
    # test with 200000 samples

    # repeat for 1000 samples
    # repeat for 100 samples
    # plot roc and indicate min_error

    iter = 1000
    # this is the test data
    test = twoClassProblem(priors, mean, cov, N = 200000)
    x_test, x_test_label = test.generateTestData()

    # train with 10000 samples
    #train = twoClassProblem(priors, mean, cov, N = 10000)
    #x_train, x_train_label =  train.generateTestData()
    #est_mu, est_covs = train.EM(iter)
    # apply the classifier on the test data
    #classify = classifier(x_test, x_test_label, priors ,est_mu, est_covs)
    #classify.minErrorClassification()

    #train = twoClassProblem(priors, mean, cov, N = 1000)
    #x_train, x_train_label =  train.generateTestData()
    #est_mu, est_covs = train.EM(iter)
    # apply the classifier on the test data
    #classify = classifier(x_test, x_test_label, priors ,est_mu, est_covs)
    #classify.minErrorClassification()

    train = twoClassProblem(priors, mean, cov, N = 100)
    x_train, x_train_label =  train.generateTestData()
    est_mu, est_covs = train.EM(iter)
    # apply the classifier on the test data
    classify = classifier(x_test, x_test_label, priors ,est_mu, est_covs)
    classify.minErrorClassification()

    

    

    

    

    
    

