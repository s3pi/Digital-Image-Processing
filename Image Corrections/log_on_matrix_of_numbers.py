import cv2
import numpy as np

def main():
	matrix = [[2, 8, 16], [32, 16, 128], [64, 32, 256]]

	matrix = np.log2(matrix)
	print("After Log Transformation\n", matrix)

	matrix = matrix - np.amin(matrix)
	matrix = matrix / float(np.amax(matrix))
	print("\n Values stretched between 0 and 1\n", matrix)
	
	matrix = matrix * 255
	print("\nAfter Normalization \n", matrix)


main()