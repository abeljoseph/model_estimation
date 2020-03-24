points_a = []
points_b = []

with open("data_files/csv/lab2_3_a.csv", 'r') as datafile_a, open("data_files/csv/lab2_3_b.csv", 'r') as datafile_b:
	for line in datafile_a:
		points_a.append(list(line))

	for line in datafile_b:
		points_b.append(list(line))


