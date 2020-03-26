import random
from math import sqrt
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt


class sequential_classifier:
	def __init__(self, A, B):
		self.A = A
		self.B = B
		self.classifier_count = 0

	# Adapted from lab 1
	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	# Adapted from lab 1
	def get_med(self, a, b, prototype_A, prototype_B):
		dist_a = sequential_classifier.get_euclidean_dist(prototype_A[0], prototype_A[1], a, b)
		dist_b = sequential_classifier.get_euclidean_dist(prototype_B[0], prototype_B[1], a, b)

		return 1 if dist_a < dist_b else 2

	def perform_classification(self, J):
		j = 1
		discriminants = []
		true_n_ab = []
		true_n_ba = []

		A = self.A
		B = self.B
		prototype_A = 0
		prototype_B = 0
		n_ba = 0  # Error count
		n_ab = 0

		while True:
			misclassified = True

			while misclassified:
				mis_A = []  # Misclassified points
				mis_B = []

				if len(A): prototype_A = A[random.randint(0, len(A) - 1)]
				if len(B): prototype_B = B[random.randint(0, len(B) - 1)]

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

			discriminants.extend([prototype_A, prototype_B])
			true_n_ab.append(n_ab)
			true_n_ba.append(n_ba)

			if (J and j > J) or (not len(A) and not len(B)):
				break

			j += 1

		return [discriminants, true_n_ab, true_n_ba]

	@staticmethod
	def classify_points(X, Y, J, discriminants, true_n_ab, true_n_ba):
		estimated_class = 0
		while J < discriminants.size:
			a_mu = discriminants[J][0,:]
			b_mu = discriminants[J][1,:]

			estimated_class = self.get_med(X, Y, a_mu, b_mu)

			if (not true_n_ab[J] and estimated_class == 1):
				break
			if (not true_n_ab[J] and estimated_class == 2):
				break
			
			J += 1
		
		return estimated_class

	def calculate_error(self, J, discriminants, true_n_ab, true_n_ba):
		K = 20
		total_error = []
		classified = 0
		for j in range(1, J):
			for k in range(1, K):
				error_rate = 0

				# Sequential Classifier
				classification = self.perform_classification(J=5)

				# Classify points in class A
				for i, pt in enumerate(self.A):
					classified = sequential_classifier.classify_points(pt[0], pt[1], J, classification[0], classification[1], classification[2])
					# Add to error rate if class A is misclassified as class B
					if classified == 2:
						error_rate += 1

				# Classify points in class B
				for i, pt in enumerate(self.B):
					classified = sequential_classifier.classify_points(pt[0], pt[1], J, classification[0], classification[1], classification[2])
					# Add to error rate if class B is misclassified as class A
					if classified == 1:
						error_rate += 1

				total_error[k] = error_rate / 400

		# a) average error rate
		average_error_rate = mean(total_error)
		# b) minimum error rate
		min_error_rate = min(total_error)
		# c) maximum error rate
		max_error_rate = max(total_error)
		# d) standard deviation of error rates
		stdev_error_rate = stdev(total_error)

		calculated_error_rates = [average_error_rate, min_error_rate, max_error_rate, stdev_error_rate]

		# Plot Error Rates
		J_vals = [1, 2, 3, 4, 5]

		plt.figure()
		plt.subplot(1, 2, 1)
		plt.title("Error Rate of Sequential Classifier as a function of J")
		plt.errorbar(J_vals, average_error_rate, stdev_error_rate, 'r', label='Avg Error Rate')
		plt.plot(J_vals, min_error_rate, 'b', label='Min Error Rate')
		plt.plot(J_vals, max_error_rate, 'g', label='Max Error Rate')
		plt.xlabel('J')
		plt.ylabel('Error Rate')
		plt.subplot(1, 2, 2)
		plt.title("Standard Deviation Error Rate of Sequential Classifier as a function of J")
		plt.plot(J_vals, stdev_error_rate, 'c', label='Stdev Error Rate')
		plt.xlabel('J')
		plt.ylabel('Error Rate')
		plt.tight_layout()
		plt.show()

		return calculated_error_rates

	def perform_estimation(self, J=1):
		if J < 1: return

		res = self.perform_classification(J)
		self.classifier_count += 1

		if J > 1:
			K = 20
			self.calculate_error(J, *res)
			return

		# J = 1

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

		# TODO: plot estimation
