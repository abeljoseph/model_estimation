import random
from math import sqrt
import numpy as np


class sequential_classifier:
	def __init__(self, A, B):
		self.A = A
		self.B = B
		self.prototype_A = A[random.randint(0, len(A)-1)]
		self.prototype_B = B[random.randint(0, len(B) - 1)]

	# Adapted from lab 1
	@staticmethod
	def get_euclidean_dist(px1, py1, px0, py0):
		return sqrt((px0 - px1) ** 2 + (py0 - py1) ** 2)

	# Adapted from lab 1
	def create_med2(self):
		num_steps = 200

		# Create Meshgrid for MED Classification
		x_grid = np.linspace(min(*self.A[:, 0], *self.B[:, 0]), max(*self.A[:, 0], *self.B[:, 0]),
							 num_steps)
		y_grid = np.linspace(min(*self.A[:, 1], *self.B[:, 1]), max(*self.A[:, 1], *self.B[:, 1]),
							 num_steps)

		x0, y0 = np.meshgrid(x_grid, y_grid)
		boundary = [[0 for _ in range(len(x_grid))] for _ in range(len(y_grid))]

		for i in range(num_steps):
			for j in range(num_steps):
				a_dist = sequential_classifier.get_euclidean_dist(self.prototype_A[0], self.prototype_A[1], x0[i][j], y0[i][j])
				b_dist = sequential_classifier.get_euclidean_dist(self.prototype_B[0], self.prototype_B[1], x0[i][j], y0[i][j])

				boundary[i][j] = a_dist - b_dist

		return [boundary, x_grid, y_grid]

	def perform_classification(self, j=1):
		discriminant = self.create_med2()
