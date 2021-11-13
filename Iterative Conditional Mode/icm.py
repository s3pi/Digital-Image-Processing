import numpy as np
import cv2
import math
from scipy.ndimage import imread
from copy import deepcopy
from os import listdir
from os.path import isfile, join


def main():
	imgpath = "/Users/3pi/Documents/DIP/Images/ICM/emma_noisy_200*180.jpg"
	target_imgpath = "/Users/3pi/Documents/DIP/Images/ICM"
	orig_img = imread(imgpath)
	orig_img = pad_image(orig_img)
	lables_26 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
	lables_18 = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255]
	lables_51 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255]
	data_constant = 1
	smoothness_prior_type = "quadratic"
	lable_vector = np.array(lables_26)
	org_img = quantize_img(orig_img, lable_vector)
	cost_orig_img = np.sum(org_img)
	img_1, cost_img_1 = restore_img(orig_img, orig_img, lable_vector, smoothness_prior_type, data_constant, cost_orig_img)
	iterations = 1
	path = join(target_imgpath, "emma_iters_" + str(iterations) + "_" + str(smoothness_prior_type) + ".jpg")
	cv2.imwrite(path, img_1)
	print(cost_orig_img)
	print(cost_img_1)

	if cost_img_1 < cost_orig_img:
		iterations = 2
		img_2, cost_img_2 = restore_img(orig_img, img_1, lable_vector, smoothness_prior_type, data_constant, cost_img_1)
		path = join(target_imgpath, "emma_iters_" + str(iterations) + "_" + str(smoothness_prior_type) + ".jpg")
		cv2.imwrite(path, img_2)

	while (iterations < 5 and cost_img_2 < cost_img_1):
		cost_img_1 = cost_img_2
		print(cost_img_1)
		iterations += 1
		img_1 = deepcopy(img_2)
		img_2, cost_img_2 = restore_img(orig_img, img_2, lable_vector, smoothness_prior_type, data_constant, cost_img_2)
		path = join(target_imgpath, "emma_iters_" + str(iterations) + "_" + str(smoothness_prior_type) + ".jpg")
		cv2.imwrite(path, img_2)
		print(cost_img_2)

def restore_img(orig_img, img_1, lable_vector, smoothness_prior_type, data_constant, cost_img_1):
	new_cost = cost_img_1
	for color in range(3):
		for i in range(1, len(orig_img) - 1):
			for j in range(1, len(orig_img[0]) - 1):
				#assumtion_1 : cost of similarity wrt pixel in the original image.
				data_cost = data_constant * (orig_img[i][j][color] - lable_vector) ** 2 
				#assumption_2 : cost of similarity wrt nearby pixels in the original image. In case of binary image, try with 0 or 1 and in case of color, try with a color pallate
				neigh_hood_indices = [img_1[i][j-1][color], img_1[i-1][j-1][color], img_1[i-1][j][color], img_1[i-1][j+1][color], img_1[i][j+1][color], img_1[i+1][j+1][color], img_1[i+1][j][color], img_1[i+1][j-1][color]]
				smoothness_cost = np.zeros(len(lable_vector))
				if smoothness_prior_type == "quadratic":
					for each in neigh_hood_indices:
						smoothness_cost += 0.5 * ((lable_vector - each) ** 2)
				elif smoothness_prior_type == "truncated":
					for each in neigh_hood_indices:
						smoothness_cost += abs(lable_vector - each)
				total_cost = data_cost + smoothness_cost
				prev_intenstity_pixel = img_1[i][j][color]
				img_1[i][j][color] = lable_vector[np.argmin(total_cost)]
				if img_1[i][j][color] < prev_intenstity_pixel:
					new_cost -= prev_intenstity_pixel
					new_cost += img_1[i][j][color]

	return img_1, new_cost

def quantize_img(img, lable_vector):
	for i in range(len(img)):
		for j in range(len(img[0])):
			for k in range(len(img[0][0])):
				img[i][j][k] = lable_vector[int(img[i][j][k] // 15)]

	return img

def pad_image(img):
	row_up = deepcopy(img[0])
	img = np.insert(img, 0, row_up, axis=0)
	row_down = deepcopy(img[-1])
	img = np.insert(img, -1, row_down, axis=0)
	col_left = deepcopy(img[:, 0])
	img = np.insert(img, 0, col_left, axis=1)
	col_right = deepcopy(img[:, -1])
	img = np.insert(img, -1, col_right, axis=1)

	return img

main()