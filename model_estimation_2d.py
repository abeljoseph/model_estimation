import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def get_covariance(set, mean):
    covariance = np.zeros((2, 2))
    for i in range(len(set)):
        temp = np.array([[set[i][0]], [set[i][1]]])
        covariance = covariance + np.matmul(temp, temp.T)

    mean_temp = np.array([[mean[0]], [mean[1]]])
    covariance = ((1/len(set)) * covariance) - np.matmul(mean_temp, mean_temp.T)
    return covariance



data_2d = scipy.io.loadmat('data_files/mat/lab2_2.mat')
al_set = np.sort(data_2d.get('al'))
bl_set = np.sort(data_2d.get('bl'))
cl_set = np.sort(data_2d.get('cl'))

x_min = min(*al_set[:, 0], *bl_set[:, 0], *cl_set[:, 0]) - 1
x_max = max(*al_set[:, 0], *bl_set[:, 0], *cl_set[:, 0]) + 1
y_min = min(*al_set[:, 1], *bl_set[:, 1], *cl_set[:, 1]) - 1
y_max = max(*al_set[:, 1], *bl_set[:, 1], *cl_set[:, 1]) + 1

x_grid = np.linspace(x_min, x_max, num=100)
y_grid = np.linspace(y_min, y_max, num=100)
x1, y1 = np.meshgrid(x_grid, y_grid)

al_mean = np.array([np.mean(al_set[:, 0]), np.mean(al_set[:, 1])])
bl_mean = np.array([np.mean(bl_set[:, 0]), np.mean(bl_set[:, 1])])
cl_mean = np.array([np.mean(cl_set[:, 0]), np.mean(cl_set[:, 1])])

al_cov = get_covariance(al_set, al_mean)
bl_cov = get_covariance(bl_set, bl_mean)
cl_cov = get_covariance(cl_set, cl_mean)

# if __name__ == '__main__':