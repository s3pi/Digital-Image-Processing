import cv2
import numpy as np

def main():
	imgpath = "/Users/3pi/Documents/DIP/Current/sample_images/kfp.jpg"
	imgpath_1 = "/Users/3pi/Documents/DIP/Current/sample_images/kfp_gamma.jpg"
	img = cv2.imread(imgpath)

	gamma = 0.5 #set gamma to some value)
	img = img / 255.
	img = img ** gamma
	img *= 255
	
	cv2.imwrite(imgpath_1, img)

main()