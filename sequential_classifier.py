import random
from math import sqrt
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
import scipy.io

classifier_count = 0


class sequential_classifier:
	def __init__(self, A, B):
		self.A = A
		self.B = B
		global classifier_count
		classifier_count += 1
		self.classifier_num = classifier_count

	# Adapted from lab 1
	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	# Adapted from lab 1
	@staticmethod
	def get_med(a, b, prototype_A, prototype_B):
		dist_a = sequential_classifier.get_euclidean_dist(prototype_A[0], prototype_A[1], a, b)
		dist_b = sequential_classifier.get_euclidean_dist(prototype_B[0], prototype_B[1], a, b)

		return 1 if dist_a < dist_b else 2

	def perform_classification(self, J=np.inf):
		j = 1
		discriminants = []
		true_n_ab = []
		true_n_ba = []

		A = self.A
		B = self.B
		prototype_A = 0
		prototype_B = 0

		while True:
			misclassified = True
			n_ba = 0  # Error count
			n_ab = 0

			while misclassified:
				mis_A = []  # Misclassified points
				mis_B = []
				n_ab, n_ba = 0, 0

				if len(A) > 0: prototype_A = A[random.randint(0, len(A) - 1)]
				if len(B) > 0: prototype_B = B[random.randint(0, len(B) - 1)]

				# Classify all points for A
				for i, pt in enumerate(A):
					res = self.get_med(pt[0], pt[1], prototype_A, prototype_B)

					if res == 2:  # Misclassified
						n_ab += 1
						mis_A.append(pt)

				# Classify all points for B
				for i, pt in enumerate(B):
					res = self.get_med(pt[0], pt[1], prototype_A, prototype_B)

					if res == 1:  # Misclassified
						n_ba += 1
						mis_B.append(pt)

				if not n_ab or not n_ba:  # No misclassified pts
					# Remove points from b that were classified as B
					if not n_ab:
						B = mis_B
					if not n_ba:
						A = mis_A
					misclassified = False

			discriminants.extend([[prototype_A, prototype_B]])
			true_n_ab.append(n_ab)
			true_n_ba.append(n_ba)

			if (j > J) or (not len(A) and not len(B)):
				break

			j += 1

		return [np.array(discriminants), true_n_ab, true_n_ba]

	@staticmethod
	def classify_points(X, Y, J, discriminants, true_n_ab, true_n_ba):
		est = 0
		while J < len(discriminants):
			a_mu = discriminants[J][0,:]
			b_mu = discriminants[J][1,:]

			est = sequential_classifier.get_med(X, Y, a_mu, b_mu)

			if not true_n_ba[J] and est == 1:
				break
			if not true_n_ab[J] and est == 2:
				break
			
			J += 1
		
		return est

	def calculate_error(self, J):
		K = 20
		average_error_rate = []
		min_error_rate = []
		max_error_rate = []
		stdev_error_rate = []
		for j in range(1, J+1):
			error_rate = np.zeros(K)
			for k in range(K):
				res = self.perform_classification(j)
				total_errors = 0
				
				classified = []
				# Classify points in class A
				for i in range(len(self.A)):
					pt = self.A[i]
					classified.append(sequential_classifier.classify_points(pt[0], pt[1], j, *res))
					# Add to error rate if class A is misclassified as class B
					if classified[i] == 2:
						total_errors += 1

				classified = []
				# Classify points in class B
				for i in range(len(self.B)):
					pt = self.B[i]
					classified.append(sequential_classifier.classify_points(pt[0], pt[1], j, *res))
					# Add to error rate if class B is misclassified as class A
					if classified[i] == 1:
						total_errors += 1

				# calcuate error rate
				error_rate[k] = (total_errors/400)
				

			print("J={}:{}".format(j,error_rate))

			# a) average error rate
			average_error_rate.append(np.average(error_rate))
			# b) minimum error rate
			min_error_rate.append(np.min(error_rate))
			# c) maximum error rate
			max_error_rate.append(np.max(error_rate))
			# d) standard deviation of error rates
			stdev_error_rate.append(np.std(error_rate))

		calculated_error_rates = [average_error_rate, min_error_rate, max_error_rate, stdev_error_rate]

		# Plot Error Rates
		J_vals = [1, 2, 3, 4, 5]

		plt.figure()
		plt.subplot(2, 1, 1)
		plt.title("Error Rate of Sequential Classifier as a function of J")
		plt.errorbar(J_vals, average_error_rate, stdev_error_rate, linestyle='-', marker='D', label='Avg Error Rate')
		plt.plot(J_vals, min_error_rate, "b.", linestyle='-', label='Min Error Rate')
		plt.plot(J_vals, max_error_rate, "r.", linestyle='-', label='Max Error Rate')
		plt.legend(loc='upper left')
		plt.xlabel('J')
		plt.ylabel('Error Rate')
		plt.subplot(2, 1, 2)
		plt.title("Standard Deviation of Error Rates of Sequential Classifier as a function of J")
		plt.plot(J_vals, stdev_error_rate, "c.",  linestyle='-', label='Stdev Error Rate')
		plt.xlabel('J')
		plt.ylabel('Standard Deviation')
		plt.tight_layout()
		plt.legend(loc='upper left')
		plt.show()

		return calculated_error_rates

	def plot_sequential(self, x, y, estimation):
		fig, ax = plt.subplots()
		ax.plot(self.A[:,0], self.A[:,1], color='b', marker='.', linestyle='', label='Class A')
		ax.plot(self.B[:,0], self.B[:,1], color='r', marker='.', linestyle='', label='Class B')
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.title(f'Classifier {self.classifier_num}')
		ax.contourf(x, y, np.matrix(estimation), colors=['#d6e9ff', '#ffb0b0'])
		ax.contour(x, y, np.matrix(estimation), colors='purple', linewidths=0.3)
		ax.legend()
		plt.show()

	def perform_estimation(self, J=np.inf):
		if J < 1: return

		res = self.perform_classification(J)

		num_steps = 100
		# Create Meshgrid for MED Classification
		x_grid = np.linspace(min(*self.A[:, 0], *self.B[:, 0]), max(*self.A[:, 0], *self.B[:, 0]),
							 num_steps)
		y_grid = np.linspace(min(*self.A[:, 1], *self.B[:, 1]), max(*self.A[:, 1], *self.B[:, 1]),
							 num_steps)

		x, y = np.meshgrid(x_grid, y_grid)
		estimation = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(len(x_grid)):
			for j in range(len(y_grid)):
				estimation[i][j] = sequential_classifier.classify_points(x[i][j], y[i][j], J, *res)

		self.plot_sequential(x, y, estimation)


data_2d = scipy.io.loadmat('data_files/mat/lab2_3.mat')
points_a = data_2d['a'].astype(float)
points_b = data_2d['b'].astype(float)

cl_1, cl_2, cl_3, cl_4 = sequential_classifier(np.array(points_a), np.array(points_b)), \
				   sequential_classifier(np.array(points_a), np.array(points_b)), \
				   sequential_classifier(np.array(points_a), np.array(points_b)), \
				   sequential_classifier(np.array(points_a), np.array(points_b))

cl_1.perform_estimation()
cl_2.perform_estimation()
cl_3.perform_estimation()
cl_4.calculate_error(J=5)

