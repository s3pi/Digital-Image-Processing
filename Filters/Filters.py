import numpy as np
import cv2
import math


def main():
	imgpath = "/Users/3pi/Documents/DIP/Images/Noise/emma_cartoon3_5_10.jpg"
	target_imgpath = "/Users/3pi/Documents/DIP/Images/Noise/emma_cartoon4_5_10.jpg"
	img = cv2.imread(imgpath)
	print(img.shape)
	kernel_size = 5
	#kernel = compute_averaging_kernel(kernel_size)
	#kernel = compute_gaussian_kernel(kernel_size)
	#convolved_image = convolution(img, kernel)
	#convolved_image = apply_bilateral(img, kernel_size)
	convolved_image = apply_carnooting_effect(img, kernel_size)
	#convolved_image = apply_nlm(img, kernel_size)
	#convolved_image = apply_zero_mean_gauss_noise(img)
	cv2.imwrite(target_imgpath, convolved_image)
	
def convolution(img, kernel):
	kernel_size = len(kernel)
	rows, cols, colors = img.shape
	target_rows = rows - kernel_size - 1
	target_cols = cols - kernel_size - 1
	convolved_image = [[[0] * 3] * target_cols] * target_rows
	convolved_image = np.asarray(convolved_image)
	for color in range(colors):
		for row in range(rows - kernel_size - 1):
			for col in range(cols - kernel_size - 1):
				patch = []
				for i in range(row, row + kernel_size):
					patch_row = []
					for j in range(col, col + kernel_size):
						patch_row.append(img[i][j][color])
					patch.append(patch_row)

				convolved_image[row][col][color] = find_dot_product(patch, kernel)

	return convolved_image

def find_dot_product(patch, kernel):#works for 2* 2 matrices
	sum = 0
	for i in range(len(patch)):
		for j in range(len(patch)):
			sum += patch[i][j] * kernel[i][j]
	return sum


def apply_zero_mean_gauss_noise(image):
  	image_shape = image.shape
  	mean = 0
  	var = 1000
  	sigma = var ** 0.5
  	gauss_random_no = np.random.normal(mean, sigma, image_shape)
  	noise = image + gauss_random_no
  	clipped_noise = np.clip(noise, 0, 255)

  	return clipped_noise

def apply_nlm(img, kernel_size):
	rows, cols, colors = img.shape
	convolved_image = np.zeros([rows, cols, colors])
	for new_img_row in range(kernel_size // 2, rows - (kernel_size // 2)):
		for new_img_col in range(kernel_size // 2, cols - (kernel_size // 2)):
			vec1_red, vec1_blue, vec1_green = compute_vectors(img, new_img_row, new_img_col, kernel_size)
			target_value_red, target_value_blue, target_value_green = compute_nlm_value_for_each_pixel(img, vec1_red, vec1_blue, vec1_green, kernel_size)

			convolved_image[new_img_row, new_img_col, 0] = target_value_red	
			convolved_image[new_img_row, new_img_col, 1] = target_value_blue
			convolved_image[new_img_row, new_img_col, 2] = target_value_green

	convolved_image = apply_min_max(convolved_image)

	return convolved_image

def compute_vectors(img, row, col, kernel_size):
	for kernel_row in range(- (kernel_size // 2), kernel_size // 2):
		for kernel_col in range(- (kernel_size // 2), kernel_size // 2):
			vector_red = []
			vector_blue = []
			vector_green = []
			vector_red.append(img[row + kernel_row][col + kernel_col][0])
			vector_blue.append(img[row + kernel_row][col + kernel_col][1])
			vector_green.append(img[row + kernel_row][col + kernel_col][2])
	
	return (vector_red, vector_blue, vector_green)

def compute_nlm_value_for_each_pixel(img, vec1_red, vec1_blue, vec1_green, kernel_size):
	rows, cols, colors = img.shape
	sigma = 5

	target_value_red = 0
	target_value_blue = 0
	target_value_green = 0
	sum_of_weight_red = 0
	sum_of_weight_blue = 0
	sum_of_weight_green = 0

	for row in range(kernel_size // 2, rows - kernel_size // 2):
		for col in range(kernel_size // 2, cols - kernel_size // 2):
			vec2_red, vec2_blue, vec2_green = compute_vectors(img, row, col, kernel_size)
			dist_red = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec1_red, vec2_red)]))
			dist_blue = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec1_blue, vec2_blue)]))
			dist_green = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec1_green, vec2_green)]))
			weight_red = compute_gaussian_func_value(dist_red, sigma)
			weight_blue = compute_gaussian_func_value(dist_blue, sigma)
			weight_green = compute_gaussian_func_value(dist_green, sigma)
			sum_of_weight_red += weight_red
			sum_of_weight_blue += weight_blue
			sum_of_weight_green += weight_green
			target_value_red += weight_red * img[row, col, 0]
			target_value_blue += weight_blue * img[row, col, 1]
			target_value_green += weight_green * img[row, col, 2]

	target_value_red=target_value_red/sum_of_weight_red #weighted normalization
	target_value_blue=target_value_blue/sum_of_weight_blue
	target_value_green=target_value_green/sum_of_weight_green

	return (target_value_red, target_value_blue, target_value_green)


def apply_carnooting_effect(img, kernel_size):
	for i in range(1):
		convolved_image = apply_bilateral(img, kernel_size)
		img = convolved_image

	return img

def apply_bilateral(img, kernel_size):
	gaussian_kernel = compute_gaussian_kernel(kernel_size)
	img = img / 255
	rows, cols, colors = img.shape
	target_rows = rows - kernel_size - 1
	target_cols = cols - kernel_size - 1
	convolved_image = [[[0] * 3] * target_cols] * target_rows
	convolved_image = np.asarray(convolved_image)
	for color in range(colors):
		for row in range(target_rows):
			for col in range(target_cols):
				patch = []
				for i in range(row, row + kernel_size):
					patch_row = []
					for j in range(col, col + kernel_size):
						patch_row.append(img[i][j][color])
					patch.append(patch_row)
				range_kernel = find_range_kernel(patch)
				dot_product_1 = find_corres_ele_product(range_kernel, gaussian_kernel)
				dot_product_1 = normalize_kernel(dot_product_1)
				convolved_image[row][col][color] = find_dot_product(patch, dot_product_1)
	
	convolved_image = apply_min_max(convolved_image)
	return convolved_image

def apply_min_max(image):
	minimum = np.amin(image)
	maximum = np.amax(image)
	image = (image - minimum) / float(maximum - minimum)
	image = image * 255

	return image

def normalize_kernel(matrix):
	sum = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			sum += matrix[i][j]
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			matrix[i][j] = matrix[i][j] / sum

	return matrix

def find_corres_ele_product(range_kernel, gaussian_kernel):
	kernel_size = len(range_kernel)
	dot_product_1 = [[0] * kernel_size] * kernel_size
	for i in range(kernel_size):
		for j in range(kernel_size):
			dot_product_1[i][j] += (range_kernel[i][j] * gaussian_kernel[i][j])

	return dot_product_1

def find_range_kernel(range_kernel):
	sigma = 10
	kernel_size = len(range_kernel)
	center_i = kernel_size // 2
	center_j = kernel_size // 2 
	for i in range(len(range_kernel)):
		for j in range(len(range_kernel)):
			range_diff = range_kernel[i][j] - range_kernel[center_i][center_j]
			range_kernel[i][j] = compute_gaussian_func_value(range_diff, sigma)

	return range_kernel

def compute_averaging_kernel(kernel_size):
	kernel = [[1/kernel_size] * kernel_size] * kernel_size
	return kernel

def compute_gaussian_kernel(kernel_size):
	sigma = 5
	center_i = kernel_size // 2
	center_j = kernel_size // 2
	kernel = [[0] * kernel_size] * kernel_size
	for i in range(kernel_size):
		for j in range(kernel_size):
			euclidean_dist = math.sqrt((center_i - i) ** 2 + (center_j - j) ** 2)
			value = compute_gaussian_func_value(euclidean_dist, sigma)
			kernel[i][j] += value

	return kernel

def compute_gaussian_func_value(x, sigma):
	a = 1 / (2 * math.pi * (sigma ** 2))
	b = math.exp(- ((x ** 2) / (2 * (sigma ** 2))))
	return a * b


main()

