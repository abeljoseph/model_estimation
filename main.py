import numpy as np
from sequential_classifier import sequential_classifier

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

	cl = sequential_classifier(points_a, points_b)
	cl.perform_classification()



if __name__ == "__main__":
	main()
