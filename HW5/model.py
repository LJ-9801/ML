import torch
import torch.nn as nn
import numpy as np
import math as m
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, neuron = 2):
        super(LogisticRegression, self).__init__()
        self.neuron = neuron
        self.bilinear1 = torch.nn.Bilinear(input_size, input_size, neuron)
        self.bilinear2 = torch.nn.Bilinear(neuron, neuron, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.bilinear1(x, x)
        out = self.bilinear2(out, out)
        out = self.sigmoid(out)
        return out

    
    

class classifier:
    def __init__(self, x, label):
        self.x = x
        self.label = label

    def SVM(self, gamma = 'auto', C = 1, test = None, testl = None, kfold = None):
        if kfold != None:
            scores = 0
            kf = KFold(n_splits=kfold)
            kf.get_n_splits(self.x.T)
            for train_index, test_index in kf.split(self.x.T):
                x_train, x_test = self.x.T[train_index], self.x.T[test_index]
                y_train, y_test = self.label[train_index], self.label[test_index]
                svm = SVC(kernel='rbf', gamma=gamma, C = C)
                svm.fit(x_train, y_train)
                y = svm.predict(x_test)
                scores += accuracy_score(y_test, y)
            avg = scores / kfold
        else:
            svm = SVC(kernel='rbf', gamma=gamma, C = C)
            svm.fit(self.x.T, self.label)
            y = svm.predict(test.T)
            avg = accuracy_score(testl.reshape(-1,1), y)
            plt.figure(1)
            plt.scatter(test[0], test[1], c=y)
            plt.title("SVM Prediction")
            plt.show()
        return avg



    def Logit(self, iter=10000, lr=0.001, neuron = 2):
        
        model = LogisticRegression(len(self.x[:,0]), 1, neuron = neuron)
        criterion = torch.nn.BCELoss(reduction='sum')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(iter):
            inputs = self.x.T
            labels = self.label.reshape(-1, 1).squeeze()

            optimizer.zero_grad()

            outputs = torch.squeeze(model(inputs))
            #print(torch.where(outputs > torch.tensor(0.5))[0].shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return model
    
    def predict(self, model, test = None, label = None, plot = False):
        test = test.T
        y_pred = model.forward(test)
        y_pred = np.where(y_pred > 0.5, 1, -1)
        accuracy = accuracy_score(label, y_pred)
        if plot == True: 
            figure = plt.figure(3)
            plt.scatter(test[np.where(y_pred == -1)[0], 0], test[np.where(y_pred == -1)[0], 1], c='r')
            plt.scatter(test[np.where(y_pred == 1)[0], 0], test[np.where(y_pred == 1)[0], 1], c='b')
            plt.title('Decision Region for 200000 samples')
            plt.legend(['Class 0', 'Class 1'])
            plt.show()
        return accuracy

    


class Data_Generation():
    def __init__(self, num_samples, priors):
        self.num_samples = num_samples
        self.priors = torch.tensor(priors)
        self.x = torch.zeros(2, num_samples)
        self.labels = torch.zeros(num_samples)

    def gen(self):
        threshold = torch.cumsum(self.priors, dim=0)
        C = len(self.priors)
        s = torch.rand(self.num_samples)
        theta = torch.rand(self.num_samples) * 2 * m.pi - m.pi
        n = torch.randn(2, self.num_samples)
        for i in range(C):
            if i == 0:
                tmp = theta[s < threshold[i]]
                new = torch.tensor([[torch.cos(i) , torch.sin(i)] for i in tmp]).T
                self.x[:, s < threshold[i]] = torch.tensor(2) * new + n[:, s < threshold[i]]
                self.labels[s < threshold[i]] = torch.tensor(i)
            else:
                tmp = theta[(s >= threshold[i-1]) & (s < threshold[i])]
                new = torch.tensor([[torch.cos(i) , torch.sin(i)] for i in tmp]).T
                self.x[:, (s >= threshold[i-1]) & (s < threshold[i])] = torch.tensor(4) * new + n[:, (s >= threshold[i-1]) & (s < threshold[i])]
                self.labels[(s >= threshold[i-1]) & (s < threshold[i])] = torch.tensor(i)
        self.labels = torch.where(self.labels == 0, -1, self.labels)
        return self.x, self.labels
    
    def plot_data(self):
        figure = plt.figure(1)
        plt.scatter(self.x[0, :], self.x[1, :], c=self.labels, s=5)
        plt.show()

            


    