import torch
import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0, F, Q, H, R):
        """
        Initializes the Kalman filter.
        
        x0: Initial state estimate.
        P0: Initial covariance estimate.
        F: State transition model.
        Q: Process noise covariance.
        H: Measurement model.
        R: Measurement noise covariance.
        """
        self.x = x0
        self.P = P0
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        
    def predict(self):
        """
        Predicts the next state and covariance estimates.
        """
        self.x = torch.matmul(self.F, self.x)
        self.P = torch.matmul(torch.matmul(self.F, self.P), torch.transpose(self.F, 0, 1)) + self.Q
        
    def update(self, z):
        """
        Updates the state and covariance estimates using the measurement z.
        """
        y = z - torch.matmul(self.H, self.x).T
        S = torch.matmul(torch.matmul(self.H, self.P), torch.transpose(self.H, 0, 1)) + self.R
        K = torch.matmul(torch.matmul(self.P, torch.transpose(self.H, 0, 1)), torch.inverse(S))
        self.x = self.x + torch.matmul(K, y.T)
        self.P = self.P - torch.matmul(torch.matmul(K, self.H), self.P)


    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Generate some noisy data
    #n = 1
    '''
    x_true = torch.zeros(n)
    y_true = torch.zeros(n)
    true_pos = torch.stack((x_true, y_true), dim=1)
    obs_pos = true_pos+torch.randn(n,2)*0.3
    '''
    # generate n sensors measurements
    std = 0.3
    sensor_pos = torch.tensor([[2,0], [0,2], [-2,0], [0,-2]])

    nsen = 4
    N = 1000
    x_true = torch.zeros(N,nsen)+torch.randn(N, nsen)*std
    y_true = torch.zeros(N,nsen)+torch.randn(N, nsen)*std
    s1_mea = torch.stack((x_true[:,0], y_true[:,0]), dim=1)
    s2_mea = torch.stack((x_true[:,1], y_true[:,1]), dim=1)
    s3_mea = torch.stack((x_true[:,2], y_true[:,2]), dim=1)
    s4_mea = torch.stack((x_true[:,3], y_true[:,3]), dim=1)

    obs_1 = s1_mea
    obs_2 = torch.mean(torch.stack((s1_mea, s2_mea), dim=1), dim=1)
    obs_3 = torch.mean(torch.stack((s1_mea, s2_mea, s3_mea), dim=1), dim=1)
    obs_4 = torch.mean(torch.stack((s1_mea, s2_mea, s3_mea, s4_mea), dim=1), dim=1)

    true_pos = torch.tensor([0,0])
    #print(x_obs.shape)

    # Initialize the Kalman filter
    x0 = torch.ones(2,1)
    P0 = torch.eye(2) * 0.25
    F = torch.eye(2) * 0.3
    Q = torch.eye(2)
    H = torch.eye(2)
    R = torch.eye(2)
    kf = KalmanFilter(x0, P0, F, Q, H, R)

    # Filter the data
    x_filt = []
    for x in obs_1:
        x_new = x
        for i in range(3):
            kf.predict()
            kf.update(x_new)
            x_new= kf.x.T
        x_filt = x_filt + [x_new]
    x_filt = torch.stack(x_filt, dim=0)
    x_filt = x_filt.squeeze()
    # Plot the results
    plt.figure(1)
    plt.scatter(sensor_pos[:,0], sensor_pos[:,1], marker="o" ,label='Sensor')
    plt.scatter(obs_1[:,0], obs_1[:,1], marker='.',label='Observed')
    plt.scatter(x_filt[:,0], x_filt[:,1], marker='*', label='Filtered')
    plt.scatter(0, 0, marker= 'x' ,label='True')
    plt.title('1 sensors')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.show()

    x_filt = []
    for x in obs_2:
        x_new = x
        for i in range(2):
            kf.predict()
            kf.update(x_new)
            x_new= kf.x.T
        x_filt = x_filt + [x_new]

    x_filt = torch.stack(x_filt, dim=0)
    x_filt = x_filt.squeeze()
    # Plot the results
    plt.figure(2)
    plt.scatter(sensor_pos[:,0], sensor_pos[:,1], marker="o" ,label='Sensor')
    plt.scatter(obs_2[:,0], obs_2[:,1], marker='.',label='Observed')
    plt.scatter(x_filt[:,0], x_filt[:,1], marker='*',label='Filtered')
    plt.scatter(0, 0, marker= 'x' ,label='True')
    plt.title('2 sensor')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.show()

    x_filt = []
    for x in obs_3:
        x_new = x
        for i in range(3):
            kf.predict()
            kf.update(x_new)
            x_new= kf.x.T
        x_filt = x_filt + [x_new]
    
    x_filt = torch.stack(x_filt, dim=0)
    x_filt = x_filt.squeeze()

    # Plot the results
    plt.figure(3)
    plt.scatter(sensor_pos[:,0], sensor_pos[:,1], marker="o" ,label='Sensor')
    plt.scatter(obs_3[:,0], obs_3[:,1],marker='.', label='Observed')
    plt.scatter(x_filt[:,0], x_filt[:,1],  marker='*',label='Filtered')
    plt.scatter(0, 0, marker= 'x' ,label='True')
    plt.title('3 sensors')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.show()



    x_filt = []
    for x in obs_4:
        x_new = x
        for i in range(3):
            kf.predict()
            kf.update(x_new)
            x_new= kf.x.T
        x_filt = x_filt + [x_new]
   
    x_filt = torch.stack(x_filt, dim=0)
    x_filt = x_filt.squeeze()
    # Plot the results
    plt.figure(4)
    plt.scatter(sensor_pos[:,0], sensor_pos[:,1], marker="o" ,label='Sensor')
    plt.scatter(obs_4[:,0], obs_4[:,1], marker='.',label='Observed')
    plt.scatter(x_filt[:,0], x_filt[:,1],  marker='*',label='Filtered')
    plt.scatter(0, 0, marker= 'x' ,label='True')
    plt.title('4 sensors')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    plt.show()