import histogram_equalization as he
import cv2
import numpy as np

def main():
	imgpath = "/Users/3pi/Documents/DIP/Images/Histogram/south_campus.jpg"
	target_imgpath = "/Users/3pi/Documents/DIP/Images/Histogram/emma.jpg"
	matched_imgpath = "/Users/3pi/Documents/DIP/Images/Histogram/south_campus_to_emma.jpg"

	img_1 = cv2.imread(imgpath)
	img_2 = cv2.imread(target_imgpath)

	for color in range(3):
		histogram_1 = he.construct_histogram(img_1, color)
		total_pixels_1 = he.compute_total_pixels(histogram_1)
		pmf_1 = he.compute_pmf(histogram_1, total_pixels_1)
		cdf_1 = he.compute_cdf(pmf_1)
		normalized_cdf_1 = he.compute_normalized_cdf(cdf_1)
		rounded_cdf_1 = he.round_off_cdf(normalized_cdf_1)
		
		
		histogram_2 = he.construct_histogram(img_2, color)
		total_pixels_2 = he.compute_total_pixels(histogram_2)
		pmf_2 = he.compute_pmf(histogram_2, total_pixels_2)
		cdf_2 = he.compute_cdf(pmf_2)
		normalized_cdf_2 = he.compute_normalized_cdf(cdf_2)
		rounded_cdf_2 = he.round_off_cdf(normalized_cdf_2)
		reversed_cdf_2 = compute_reverse_cdf_2(rounded_cdf_2)

		transform_img(img_1, color, rounded_cdf_1, reversed_cdf_2)

	cv2.imwrite(matched_imgpath, img_1)

def compute_reverse_cdf_2(rounded_cdf_2):
	reversed_cdf_2 = {}
	for i in range(len(rounded_cdf_2)):
		if rounded_cdf_2[i] not in reversed_cdf_2.keys():
			reversed_cdf_2.update({rounded_cdf_2[i] : i})
	return reversed_cdf_2

def transform_img(img_1, color, rounded_cdf_1, reversed_cdf_2):
	min_reversed_cdf_2 = min(reversed_cdf_2.keys())
	max_reversed_cdf_2 = max(reversed_cdf_2.keys())
	for j in range(len(img_1[0])):
		for i in range(len(img_1)):
			closest_intensity = rounded_cdf_1[img_1[i][j][color]]
			if (closest_intensity not in reversed_cdf_2.keys()):
				if closest_intensity < min_reversed_cdf_2:
					closest_intensity = min_reversed_cdf_2
				elif closest_intensity > max_reversed_cdf_2:
					closest_intensity = max_reversed_cdf_2
				else:
					left_steps = 0
					right_steps = 0
					while((closest_intensity + right_steps) not in reversed_cdf_2.keys()):
						right_steps += 1
					while((closest_intensity - left_steps) not in reversed_cdf_2.keys()):
						left_steps -= 1
					if right_steps > left_steps:
						closest_intensity -= left_steps
					else:
						closest_intensity += right_steps

			img_1[i][j][color] = reversed_cdf_2[closest_intensity]

if __name__ == '__main__':
	main()





