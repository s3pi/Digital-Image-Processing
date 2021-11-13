import numpy as np
import matplotlib.pyplot as plt
import cv2
import histogram_equalization as he

imgpath = "/Users/3pi/Documents/DIP/Images/Histogram/dragon_and_dany.jpg"
imgpath_resultant = "/Users/3pi/Documents/DIP/Images/Histogram/dragon_and_dany_hist_equalization.jpg"
img = cv2.imread(imgpath)

def main():
	plot_histogram(img)
	#img_copy = deepcopy(img)
	he.histogram_equalization(img)
	plot_histogram(img)

def plot_histogram(img):
	histogram = []
	for color in range(len(img[0][0])):
		for j in range(len(img[0])):
			for i in range(len(img)):
				histogram.append(img[i][j][color])

	n, bins, patches = plt.hist(histogram)
	plt.axis([0, 255, 0, 600000])
	plt.show()

main()