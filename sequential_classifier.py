import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


class sequential_classifier:
	def __init__(self, A, B):
		self.A = A
		self.B = B
		self.prototype_A = A[random.randint(0, len(A) - 1)]
		self.prototype_B = B[random.randint(0, len(B) - 1)]

	# Adapted from lab 1
	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	# Adapted from lab 1
	def get_med(self, a, b):
		dist_a = sequential_classifier.get_euclidean_dist(self.prototype_A[0], self.prototype_A[1], a, b)
		dist_b = sequential_classifier.get_euclidean_dist(self.prototype_B[0], self.prototype_B[1], a, b)

		return 1 if dist_a < dist_b else 2

	def perform_classification(self, j=1):
		misclassified = True
		mis_A = []  # Misclassified points
		mis_B = []

		while misclassified:
			n_ba = 0  # Error count
			n_ab = 0

			# Classify all points for A
			for i, pt in enumerate(self.A):
				res = self.get_med(pt[0], pt[1])

				if res == 2:  # Misclassified
					n_ab += 1
					mis_A.append(pt)

			# Classify all points for B
			for i, pt in enumerate(self.B):
				res = self.get_med(pt[0], pt[1])

				if res == 1:  # Misclassified
					n_ba += 1
					mis_B.append(pt)

			if not n_ab or not n_ba:  # No misclassified pts
				misclassified = False

