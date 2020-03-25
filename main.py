import numpy as np 

def main():
	# Section 3
	points_a = []
	points_b = []

	with open("data_files/csv/lab2_3_a.csv", 'r') as datafile_a, open("data_files/csv/lab2_3_b.csv", 'r') as datafile_b:
		for line in datafile_a:
			points_a.append([int(x) for x in line.split(',')])

		for line in datafile_b:
			points_b.append([int(x) for x in line.split(',')])

	points_a = np.array(points_a)
	points_b = np.array(points_b)


	num_steps = 100

	# Create Meshgrid for MED Classification
	x_grid = np.linspace(min(*points_a[:, 0], *points_b[:, 0]), max(*points_a[:, 0], *points_b[:, 0]),
						 num_steps)
	y_grid = np.linspace(min(*points_a[:, 1], *points_b[:, 1]), max(*points_a[:, 1], *points_b[:, 1]),
						 num_steps)


if __name__ == "__main__":
	main()
