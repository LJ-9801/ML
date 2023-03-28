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

    problem = twoClassProblem(priors, mean, cov, N = 20000)
    x,label = problem.generateTestData()
    problem.plotTrueClass()

    mean = [meam_01, meam_02, mean1]
    cov = [cov_01, cov_02, cov1]

    classify = classifier(x, label, priors, mean, cov)
    classify.minErrorClassification()