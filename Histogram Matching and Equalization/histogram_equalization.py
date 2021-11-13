import cv2
import numpy as np
 
def main():
	print(cv2.__version__)
	imgpath = "/Users/3pi/Documents/DIP/Images/Histogram/dragon_and_dany.jpg"
	imgpath_resultant = "/Users/3pi/Documents/DIP/Images/Histogram/dragon_and_dany_hist_equalization.jpg"
	img = cv2.imread(imgpath) #reading the image from the path and storing it as an array (length * breadth * color)
	#cv2.imshow('image', img) #figure this out
	print(img.shape) #prints the 3 tuple - (height, depth, color - 3 for RBG, 2 - for gray, 3 component is 0)
	#histogram = cv2.calcHist([img],[0],None,[256],[0,256]) #path, gray scale channel, no mask, full size, build histogram for all pixel intensities
	histogram_equalization(img)
	cv2.imwrite(imgpath_resultant, img)

def histogram_equalization(img):
	for color in range(len(img[0][0])):
		histogram = construct_histogram(img, color)
		total_pixels = compute_total_pixels(histogram)
		pmf = compute_pmf(histogram, total_pixels)
		cdf = compute_cdf(pmf)
		normalized_cdf = compute_normalized_cdf(cdf)
		rounded_cdf = round_off_cdf(normalized_cdf)
		#print("cdf is saturating at ", rounded_cdf[255]) #testing if cdf is saturating at 1.
		transform_img(img, color, rounded_cdf)

def transform_img(img, color, rounded_cdf):
	for j in range(len(img[0])):
		for i in range(len(img)):
			img[i][j][color] = rounded_cdf[img[i][j][color]]


def round_off_cdf(normalized_cdf):
	rounded_cdf = []
	for i in range(len(normalized_cdf)):
		rounded_cdf.append(round(normalized_cdf[i]))
	return rounded_cdf

def compute_normalized_cdf(cdf):
	normalized_cdf = []
	for i in range(len(cdf)):
		normalized_cdf.append(cdf[i] * 255)
	return normalized_cdf

def compute_cdf(pmf):
	cdf = []
	cdf.append(pmf[0])
	for i in range(1, len(pmf)):
		cdf.append(cdf[i - 1] + pmf[i])
	return cdf

def compute_pmf(histogram, total_pixels):
	pmf = []
	for i in range(len(histogram)):
		pmf.append(histogram[i]/total_pixels)
	return pmf

def compute_total_pixels(histogram):
	total_pixels = 0
	for i in histogram:
		total_pixels += i
	return total_pixels

def construct_histogram(img, color):
	histogram = [0] * 256
	for j in range(len(img[0])):
		for i in range(len(img)):
			histogram[img[i][j][color]] += 1
	return histogram

if __name__ == '__main__':
	main()