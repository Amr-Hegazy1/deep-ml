def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.

	n = len(a)

	m = len(a[0])

	if(m != len(b)):
		return -1

	res = []

	for i, row in enumerate(a):
		tmp = 0
		for j, col in enumerate(a[i]):
			tmp += col * b[j]
		res.append(tmp)

	return res