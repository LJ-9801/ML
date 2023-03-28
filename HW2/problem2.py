import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal



class threeClassProblem:
    def __init__(self, priors, mean, cov, N):
        self.priors = priors
        self.mean = mean
        self.cov = cov
        self.N = N

    # generate data
    def generateTestData(self):

        # generate a mixture of len(priors) Gaussian distributions
        def generateMixData(N, priors, mean, cov):
            C = len(priors) # number of classes
            n = len(mean[0]) # data dimension
            x = np.zeros((n, N))
            label = np.zeros(N)
            u = np.random.rand(N)
            threshold = np.cumsum(priors)
            for i in range(C):
                if i == 0:
                    idx = np.where(u <= threshold[i])[0]
                else:  
                    idx = np.where((u > threshold[i-1]) & (u <= threshold[i]))[0]
                x[:, idx] = np.random.multivariate_normal(mean[i], cov[i], len(idx)).T
                label[idx] = i
            return x.T

        self.Classes = len(self.priors) # number of classes
        self.n = len(self.mean[0]) # data dimension
        self.x = np.zeros((self.n, self.N))
        self.label = np.zeros(self.N)

        u = np.random.rand(self.N)
        threshold = np.cumsum(self.priors)

        for i in range(self.Classes):
            if i == 0:
                idx = np.where(u <= threshold[i])[0]
                self.x[:, idx] = np.random.multivariate_normal(mean[i], cov[i], len(idx)).T
            elif i == 1:  
                idx = np.where((u > threshold[i-1]) & (u <= threshold[i]))[0]
                self.x[:, idx] = np.random.multivariate_normal(mean[i], cov[i], len(idx)).T
            else:
                idx = np.where((u > threshold[i-1]) & (u <= threshold[i]))[0]
                mean2 = [mean[2], mean[3]]
                cov2 = [cov[2], cov[3]]
                w2 = [0.5, 0.5]
                self.x[:, idx] = generateMixData(len(idx), w2, mean2, cov2).T
            self.label[idx] = i
        
        self.x_class0 = self.x[:, np.where(self.label == 0)[0]]
        self.x_class1 = self.x[:, np.where(self.label == 1)[0]]
        self.x_class2 = self.x[:, np.where(self.label == 2)[0]]
            
        # p_class2 and data2 are class 1 and class 2, data3 and data4 are class 3
        # data3 and data4 are of equal weights which is 1/2
        # pdf_class1 = data1
        # pdf_class2 = data2
        # class prior is 0.3, 0.3, 0.4
        

    # plot sample with true labels
    def plotTrueClass(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(projection='3d')
        #print(self.x_class0.shape)
        c1 = ax.scatter(self.x_class0[0, :], self.x_class0[1, :], self.x_class0[2, :], c='r', marker='x')
        c2 = ax.scatter(self.x_class1[0, :], self.x_class1[1, :], self.x_class1[2, :], c='g', marker='^')
        c3 = ax.scatter(self.x_class2[0, :], self.x_class2[1, :], self.x_class2[2, :], c='b', marker='o')
        plt.title('True Classification')
        ax.legend(handles = [c1,c2,c3], labels = ['Class 1', 'Class 2', 'Class 3'])
        plt.show()

    # perform LDA analysis
    def chooseClass(self, loss_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]):

        def risk(i,x,loss,mean,cov,priors):
            risk = 0
            for j in range(len(priors)):
                if j != i:
                    risk = risk + priors[j]*loss[i][j]*multivariate_normal.pdf(x, mean[j], cov[j])
            return risk
        
        self.predictions = np.zeros(self.N)
        for i in range(len(self.x[0])):
            pred = np.argmin([risk(j, self.x[:,i], loss_matrix, self.mean, self.cov, self.priors) for j in range(len(self.priors))])
            self.predictions[i] = pred
       
        # now we implement ERM classifier to make decision
        # using 0-1 loss function

    def plotPrediction(self):
        fig = plt.figure(2)
        ax = fig.add_subplot(projection='3d')
        x_class0 = self.x[:, np.where(self.predictions == 0)[0]]
        x_class1 = self.x[:, np.where(self.predictions == 1)[0]]
        x_class2 = self.x[:, np.where(self.predictions == 2)[0]]
        c1 = ax.scatter(x_class0[0, :], x_class0[1, :], x_class0[2, :], c='r', marker='x')
        c2 = ax.scatter(x_class1[0, :], x_class1[1, :], x_class1[2, :], c='g', marker='^')
        c3 = ax.scatter(x_class2[0, :], x_class2[1, :], x_class2[2, :], c='b', marker='o')
        ax.legend(handles = [c1,c2,c3], labels = ['Pred 1', 'Pred 2', 'Pred 3'])
        plt.title('Predicted Classification')
        plt.show()

    # plot the point that is misclassified vs truely classified
    def plotMisclassified(self):
        fig = plt.figure(3)
        ax = fig.add_subplot(projection='3d')
        #print(np.intersect1d(np.where(self.predictions == 0)[0], np.where(self.label == 0)[0]))
        x_class1T = self.x[:, np.where((self.predictions == 0) & (self.label == 0))[0]]
        x_class1F = self.x[:, np.where((self.predictions == 0) & (self.label != 0))[0]]

        x_class2T = self.x[:, np.where((self.predictions == 1) & (self.label == 1))[0]]
        x_class2F = self.x[:, np.where((self.predictions == 1) & (self.label != 1))[0]]

        x_class3T = self.x[:, np.where((self.predictions == 2) & (self.label == 2))[0]]
        x_class3F = self.x[:, np.where((self.predictions == 2) & (self.label != 2))[0]]

        c1T = ax.scatter(x_class1T[0, :], x_class1T[1, :], x_class1T[2, :], c='g', marker='x')
        c1F = ax.scatter(x_class1F[0, :], x_class1F[1, :], x_class1F[2, :], c='r', marker='x')

        c2T = ax.scatter(x_class2T[0, :], x_class2T[1, :], x_class2T[2, :], c='g', marker='^')
        c2F = ax.scatter(x_class2F[0, :], x_class2F[1, :], x_class2F[2, :], c='r', marker='^')

        c3T = ax.scatter(x_class3T[0, :], x_class3T[1, :], x_class3T[2, :], c='g', marker='o')
        c3F = ax.scatter(x_class3F[0, :], x_class3F[1, :], x_class3F[2, :], c='r', marker='o')
        ax.legend(handles = [c1T,c2T,c3T], labels = ['Label 1', 'Label 2', 'Label 3'])
        plt.title('Correctly Classified(Green) vs Misclassified(Red)')
        plt.show()

    def plotConfusionMatrix(self):
        confusion_matrix = np.zeros((3,3))
        for i in range(self.N):
            confusion_matrix[int(self.predictions[i])][int(self.label[i])] += int(1)
        print(confusion_matrix)

if __name__ == "__main__":
    mean1 = [0, 0, 0] #class 1
    mean2 = [1, 1, 1] # class 2

    #class 3
    mean3 = [0.5, 0, 0]
    mean4 = [0, 0, -0.5]

    

    cov1 = [[3, 0, 0], 
            [0, 3, 0], 
            [0, 0, 3]]

    cov2 = [[0.5, 0, 0],
            [0, 1, 0],
            [0, 0, 2]]

    cov3 = [[2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]]

    cov4 = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]

    # this model will take 
    # mean 1, cov 1 as class 0
    # mean 2, cov 2 as class 1
    # mean (3,4), cov (3,4) as class 2

    mean = [mean1, mean2, mean3, mean4]
    cov = [cov1, cov2, cov3, cov4]
    priors = [0.3, 0.3, 0.4]

    L10 = [[0 , 1, 10],
           [1, 0, 10],
           [1, 1, 0]]

    L100 = [[0 , 1, 100],
           [1, 0, 100],
           [1, 1, 0]]

    problem = threeClassProblem(priors, mean, cov, N=10000)
    problem.generateTestData()
    #problem.plotTrueClass()

    #problem.chooseClass()
    #problem.chooseClass(L10)
    problem.chooseClass(L100)

    problem.plotPrediction()
    problem.plotMisclassified()
    problem.plotConfusionMatrix()


