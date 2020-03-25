import scipy.io
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':

    print("Min X: ", x_min)
    print("Max X: ", x_max)
    print("Min Y: ", y_min)
    print("Max Y: ", y_max)
