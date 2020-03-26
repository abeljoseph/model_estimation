import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def get_gaussian(mean, sd, x):
    return (1/(np.sqrt(2*np.pi)*sd)) * np.exp([-1/2*(((i-mean)/sd)**2) for i in x])


def get_exponential(_lambda, x):
    return _lambda * np.exp([-_lambda*i for i in x])


def get_uniform(x):
    return [1/(max(x)-min(x)) for i in range(0,len(x))]


def plot_comparison(a_pdf_estimated, b_pdf_estimated, estimation_type):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot a
    axs[0].set_title("{} of a".format(estimation_type))
    axs[0].set(xlim=(0, 10), ylim=(0, 1))
    axs[0].plot(a_vector, a_pdf_true, label='True p(x)')
    axs[0].plot(a, a_pdf_estimated, label='Estimated p(x)')

    # Plot b
    axs[1].set_title("{} of b".format(estimation_type))
    axs[1].set(xlim=(0, 5), ylim=(0, 1))
    axs[1].plot(b_vector, b_pdf_true, label='True p(x)')
    axs[1].plot(b, b_pdf_estimated, label='Estimated p(x)')

    for ax in axs:
        ax.set(xlabel='x', ylabel='p(x)')
        ax.legend(loc='upper right')
        ax.grid()


def plot_parzen(a_pdf_estimated_1, a_pdf_estimated_2, b_pdf_estimated_1, b_pdf_estimated_2, estimation_type_1, estimation_type_2):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot a 1
    axs[0][0].set_title("{} of a".format(estimation_type_1))
    axs[0][0].set(xlim=(0, 10), ylim=(0, 1))
    axs[0][0].plot(a_vector, a_pdf_estimated_1, label='Estimated p(x)')
    axs[0][0].plot(a_vector, a_pdf_true, label='True p(x)')

    # Plot b 1
    axs[0][1].set_title("{} of b".format(estimation_type_1))
    axs[0][1].set(xlim=(0, 5), ylim=(0, 1))
    axs[0][1].plot(b_vector, b_pdf_estimated_1, label='Estimated p(x)')
    axs[0][1].plot(b_vector, b_pdf_true, label='True p(x)')

    # Plot a 2
    axs[1][0].set_title("{} of a".format(estimation_type_2))
    axs[1][0].set(xlim=(0, 10), ylim=(0, 1))
    axs[1][0].plot(a_vector, a_pdf_estimated_2, label='Estimated p(x)')
    axs[1][0].plot(a_vector, a_pdf_true, label='True p(x)')

    # Plot b 2
    axs[1][1].set_title("{} of b".format(estimation_type_2))
    axs[1][1].set(xlim=(0, 5), ylim=(0, 1))
    axs[1][1].plot(b_vector, b_pdf_estimated_2, label='Estimated p(x)')
    axs[1][1].plot(b_vector, b_pdf_true, label='True p(x)')

    for ax in axs:
        for axis in ax:
            axis.set(xlabel='x', ylabel='p(x)')
            axis.legend(loc='upper right')
            axis.grid()


def parametric_gaussian():
    a_mean = np.mean(a)
    a_sd = np.std(a)
    a_pdf_estimated = get_gaussian(a_mean, a_sd, a)

    b_mean = np.mean(b)
    b_sd = np.std(b)
    b_pdf_estimated = get_gaussian(b_mean, b_sd, b)

    plot_comparison(a_pdf_estimated, b_pdf_estimated, "Parametric Estimation - Gaussian")


def parametric_exponential():
    a_rate = 1/np.mean(a)
    a_pdf_estimated = get_exponential(a_rate, a)

    b_rate = 1/np.mean(b)
    b_pdf_estimated = get_exponential(b_rate, b)
    
    plot_comparison(a_pdf_estimated, b_pdf_estimated, "Parametric Estimation - Exponential")


def parametric_uniform():
    a_pdf_estimated = get_uniform(a)

    b_pdf_estimated = get_uniform(b)

    plot_comparison(a_pdf_estimated, b_pdf_estimated, "Parametric Estimation - Uniform")


def non_parameteric():
    parzen_sd_1 = 0.1
    parzen_sd_2 = 0.4

    a_pdf_estimated_1 = [sum(get_gaussian(i, parzen_sd_1, a))/a_size for i in a_vector]
    a_pdf_estimated_2 = [sum(get_gaussian(i, parzen_sd_2, a))/a_size for i in a_vector]

    b_pdf_estimated_1 = [sum(get_gaussian(i, parzen_sd_1, b))/b_size for i in b_vector]
    b_pdf_estimated_2 = [sum(get_gaussian(i, parzen_sd_2, b))/b_size for i in b_vector]

    plot_parzen(a_pdf_estimated_1, a_pdf_estimated_2, b_pdf_estimated_1, b_pdf_estimated_2, "Non-Parametric Estimation - Gaussian (σ=0.1)", "Non-Parametric Estimation - Gaussian (σ=0.4)")


data_1d = scipy.io.loadmat('data_files/mat/lab2_1.mat')

# a dataset values
a = np.sort(data_1d.get('a')[0])
a_mu = 5
a_sigma = 1
a_size = len(a)
a_vector = np.linspace(min(a), max(a), num=100)
a_pdf_true = get_gaussian(a_mu, a_sigma, a_vector)

# b dataset values
b = np.sort(data_1d.get('b')[0])
b_lambda = 1
b_size = len(b)
b_vector = np.linspace(min(b), max(b), num=100)
b_pdf_true = get_exponential(b_lambda, b_vector)
    

if __name__ == '__main__':
    parametric_gaussian()
    parametric_exponential()
    parametric_uniform()
    non_parameteric()

    plt.show()
