import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

def get_covariance(set, mean):
    covariance = np.zeros((2, 2))
    # for i in range(len(set)):
    for i in range(1):
        temp = [[set[i][0], set[i][1]]]
        print(temp)
        print(np.array(temp).T)
        covariance = np.matmul(np.array(temp).T, temp)
        print(covariance)

    mean_temp = np.array([[mean[0]], [mean[1]]])
    covariance = ((1/len(set)) * covariance) - np.matmul(mean_temp, mean_temp.T)
    return covariance


def get_ML_pair_boundary(Sa, Sb, Ma, Mb, x, y):
    num_steps = len(x)

    boundary = [[0 for _ in range(num_steps)] for _ in range(num_steps)]

    inv_cov_a = np.linalg.inv(Sa)
    inv_cov_b = np.linalg.inv(Sb)
    mean_a = np.array(Ma)
    mean_b = np.array(Mb)

    Q0 = np.subtract(inv_cov_a, inv_cov_b)
    Q1 = 2 * (np.matmul(mean_b,inv_cov_b) - np.matmul(mean_a,inv_cov_a))
    Q2 = np.matmul(np.matmul(mean_a, inv_cov_a), mean_a.T) - np.matmul(np.matmul(mean_b, inv_cov_b), mean_b.T)

    for i in range(num_steps):
        for j in range(num_steps):
            coord = [[x[i][j]], [y[i][j]]]
            dist = np.matmul(np.matmul(np.array(coord).T, Q0), coord) + np.matmul(Q1, coord) + Q2
            boundary[i][j] = dist[0][0]

    return boundary


def get_ML_boundary(x, y, ML_ab, ML_ac, ML_bc):
    num_steps = len(x)

    boundary = [[0 for _ in range(num_steps)] for _ in range(num_steps)]

    for i in range(num_steps):
        for j in range(num_steps):
            if ML_ab[i][j] >= 0 and ML_bc[i][j] <= 0:
                boundary[i][j] = 2
            elif ML_bc[i][j] >= 0 and ML_ac[i][j] >= 0:
                boundary[i][j] = 3
            elif ML_ac[i][j] <= 0 and ML_ab[i][j] <= 0:
                boundary[i][j] = 1

    return boundary


def plot_parametric(boundary, x, y, al, bl, cl):
    plt.title("Parametric Estimation of al, bl, cl")
    plt.ylabel("Feature 2")
    plt.xlabel("Feature 1")

    plt.scatter(al[:, 0], al[:, 1], color='r')
    plt.scatter(bl[:, 0], bl[:, 1], color='g')
    plt.scatter(cl[:, 0], cl[:, 1], color='b')

    contour = plt.contour(x, y, boundary, colors="purple")

    handles = [Rectangle((0, 0), 1, 1, color='r'), Rectangle((0, 0), 1, 1, color='g'),
               Rectangle((0, 0), 1, 1, color='b'), contour.collections[0]]
    labels = ['Class al', 'Class bl', 'Class cl', 'ML Classifier']

    plt.legend(handles, labels)

    plt.show()


data_2d = scipy.io.loadmat('data_files/mat/lab2_2.mat')
al_set = data_2d.get('al')
bl_set = data_2d.get('bl')
cl_set = data_2d.get('cl')

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
# bl_cov = get_covariance(bl_set, bl_mean)
# cl_cov = get_covariance(cl_set, cl_mean)
#
# ML_ab = get_ML_pair_boundary(al_cov, bl_cov, al_mean, bl_mean.T, x1, y1)
# ML_ac = get_ML_pair_boundary(al_cov, cl_cov, al_mean, cl_mean.T, x1, y1)
# ML_bc = get_ML_pair_boundary(bl_cov, cl_cov, bl_mean, cl_mean.T, x1, y1)
#
# total_boundary_plot = get_ML_boundary(x1, y1, ML_ab, ML_ac, ML_bc)

# if __name__ == '__main__':
    # plot_parametric(total_boundary_plot, x_grid, y_grid, al_set, bl_set, cl_set)