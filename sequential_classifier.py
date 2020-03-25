import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


class sequential_classifier:
	def __init__(self, A, B):
		self.A = A
		self.B = B

	# Adapted from lab 1
	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	# Adapted from lab 1
	def get_med(self, a, b, prototype_A, prototype_B):
		dist_a = sequential_classifier.get_euclidean_dist(prototype_A[0], prototype_A[1], a, b)
		dist_b = sequential_classifier.get_euclidean_dist(prototype_B[0], prototype_B[1], a, b)

		return 1 if dist_a < dist_b else 2

	def perform_classification(self, J=0):
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

			if (J and j > J) or (not A and not B):
				break

			j += 1

		return [discriminants, true_n_ab, true_n_ba]
