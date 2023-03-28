import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class classifier:
    def __init__(self, x, label, priors, mean, cov):
        self.x = x
        self.label = label
        self.priors = priors
        self.mean = mean
        self.cov = cov
        self.x_class0 = x[:, label == 0]
        self.x_class1 = x[:, label == 1]

    def minErrorClassification(self):
        
        gamma = np.arange(0, 100, 0.1)

        #gamma = np.log(gamma, dtype=np.float64)
        roc = np.zeros((2, len(gamma)))
        p_error = np.zeros(len(gamma))
        the_error = None
        for th in range(len(gamma)):
            #multivariate_normal.pdf(x, self.mean[0][0], self.cov[0][0])
            pred = np.divide(multivariate_normal.pdf(self.x.T, self.mean[2], self.cov[2]), 
                             0.5*multivariate_normal.pdf(self.x.T, self.mean[0], self.cov[0])+ 0.5*multivariate_normal.pdf(self.x.T, self.mean[1], self.cov[1])) >= gamma[th]
            #print(pred)
            
            correct = np.where(pred == self.label)[0]
            incorrect = np.where(pred != self.label)[0]

            TP = np.where(pred[correct] == 1)[0].shape[0]
            FP = np.where(pred[incorrect] == 1)[0].shape[0]
            FN = np.where(pred[incorrect] == 0)[0].shape[0]
            TN = np.where(pred[correct] == 0)[0].shape[0]
            
            #confusion = np.array([[TP, FP], [FN, TN]])
            
            roc[0, th] = FP/(FP+TN)
            roc[1, th] = TP/(TP+FN)

            p_error[th] = roc[0, th]*self.priors[0] + (1-roc[1, th])*self.priors[1]

            if(abs(gamma[th] - 0.66) <= 0.05):
                the_error = th


        # find the point with theroetical error 0.67
        fig = plt.figure(2)
        print('min error = ', np.min(p_error))
        min_error = np.argmin(p_error)
        plt.scatter(roc[0, the_error], roc[1, the_error], c='g', marker='o')
        plt.scatter(roc[0, min_error], roc[1, min_error], c='r', marker='o')
        plt.plot(roc[0, :], roc[1, :])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(['Theoretical Error','Min Error','ROC'])
        plt.title('ROC')
        plt.show()
    


    def Logit(self, iter=10000, lr=0.001, bilinear=False, test = None):
        class LogisticRegression(torch.nn.Module):
            def __init__(self, input_size, output_size):
                super(LogisticRegression, self).__init__()
                self.bilinear = bilinear
                if bilinear == False:
                    self.linear = torch.nn.Linear(input_size, output_size)
                else:
                    self.bilinear = torch.nn.Bilinear(input_size, input_size, output_size)

            def forward(self, x):
                if self.bilinear == False:
                    out = torch.sigmoid(self.linear(x))
                else:
                    out = torch.sigmoid(self.bilinear(x, x))
                return out
        
        model = LogisticRegression(len(self.mean[0]), 1)
        criterion = torch.nn.BCELoss(reduction='sum')

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for epoch in range(iter):
            inputs = torch.from_numpy(self.x.T).float()
            labels = torch.from_numpy(self.label.T).float()

            optimizer.zero_grad()

            outputs = torch.squeeze(model(inputs))
            #print(torch.where(outputs > torch.tensor(0.5))[0].shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if test.all() != None:
            test = torch.from_numpy(test.T).float()
            y_pred = model.forward(test)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            figure = plt.figure(3)
            plt.scatter(test[np.where(y_pred == 0)[0], 0], test[np.where(y_pred == 0)[0], 1], c='r')
            plt.scatter(test[np.where(y_pred == 1)[0], 0], test[np.where(y_pred == 1)[0], 1], c='b')
            plt.title('Decision Region for 200000 samples')
            plt.legend(['Class 0', 'Class 1'])
            plt.show()
            return y_pred
        else:    
            y_pred = model.forward(torch.from_numpy(self.x.T).float())
            y_pred = np.where(y_pred > 0.5, 1, 0)
            figure = plt.figure(3)
            plt.scatter(self.x[0, :], self.x[1, :], c=y_pred)
            plt.title('Decision Region for 200000 samples')
            plt.legend(['Class 0', 'Class 1'])
            plt.show()
            return y_pred
        







class twoClassProblem:
    def __init__(self, priors, mean, cov, N):
        self.priors = priors
        self.mean = mean
        self.cov = cov
        self.N = N
        self.x = None

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
                mean1 = self.mean[0]
                cov1 = self.cov[0]
                w1 = [0.5, 0.5]
                self.x[:, idx] = generateMixData(len(idx), w1, mean1, cov1).T
            elif i == 1:  
                idx = np.where((u > threshold[i-1]) & (u <= threshold[i]))[0]
                self.x[:, idx] = np.random.multivariate_normal(self.mean[i], self.cov[i], len(idx)).T
            self.label[idx] = i
        
        self.x_class0 = self.x[:, np.where(self.label == 0)[0]]
        self.x_class1 = self.x[:, np.where(self.label == 1)[0]]

        return self.x, self.label

    # plot sample with true labels
    def plotTrueClass(self):
        fig = plt.figure(1)
        #print(self.x_class0.shape)
        c1 = plt.scatter(self.x_class0[0, :], self.x_class0[1, :], c='r', marker='x')
        c2 = plt.scatter(self.x_class1[0, :], self.x_class1[1, :], c='g', marker='^')
        plt.title('True Classification')
        fig.legend(handles = [c1,c2], labels = ['Class 0', 'Class 1'])
        plt.show()


    def Estep(self, mu, covs, priors):

        h = MultivariateNormal(mu, covs)
        ll = h.log_prob(self.x)
        w = ll + priors
        sumll = torch.logsumexp(w, dim=1, keepdim=True)
        log_posterior = w - sumll

        return log_posterior
        
    def Mstep(self, log_posterior, iter, classes):
        eps = torch.finfo(torch.float32).eps
        pi = torch.exp(log_posterior.reshape(iter, classes, 1))
        
        pi = pi*(1-classes*eps) + eps     
        mu = torch.sum(pi*self.x, dim=0)/torch.sum(pi, dim=0)

        delta = pi*(self.x - mu)
        covs = torch.matmul(delta.permute(1,2,0), delta.permute(1,0,2))/torch.sum(pi, dim=0).reshape(classes, 1, 1)

        return mu, covs

    
    def EM(self, max_iter=500):
        # for class 0, we have a mixture of two Gaussian
        # we first make a guess of the two means and covariances
        # then we use EM algorithm to estimate the parameters
        # first we make a guess of the two means and covariances
        # mu1, sigma1, mu2, sigma2
        
        classes = 3
        dim = 2
        self.x = self.x.T.reshape(self.N,1,2)
        self.x  = torch.tensor(self.x)                           
        mu = torch.randn(classes, dim)
        covs = torch.stack(classes*[torch.eye(dim)])
        priors = torch.tensor([0.3,0.3,0.4]).log() # we take the log of the priors

        for i in range(max_iter):
            log_posterior = self.Estep(mu, covs, priors)
            mu, covs = self.Mstep(log_posterior, self.N, classes)

        
        m01 = abs(mu - torch.tensor(self.mean[0][0]))
        m01 = torch.sum(m01, dim=1)
        m01_idx = torch.argmin(m01)
        m01 = mu[m01_idx]
        cov01 = covs[m01_idx]

        m02 = abs(mu - torch.tensor(self.mean[0][1]))
        m02 = torch.sum(m02, dim=1)
        m02_idx = torch.argmin(m02)
        m02 = mu[m02_idx]
        cov02 = covs[m02_idx]

        m1 = abs(mu - torch.tensor(self.mean[1]))
        m1 = torch.sum(m1, dim=1)
        m1_idx = torch.argmin(m1)
        m1 = mu[m1_idx]
        cov1 = covs[m1_idx]

        mu = torch.stack([m01, m02, m1])
        covs = torch.stack([cov01, cov02, cov1])
        
        return mu, covs