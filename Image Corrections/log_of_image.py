import cv2
import numpy as np

def main():
	imgpath = "/Users/3pi/Documents/DIP/Current/sample_images/sunrise.jpg"
	imgpath_1 = "/Users/3pi/Documents/DIP/Current/sample_images/sunrise_log.jpg"
	img = cv2.imread(imgpath)

	img = np.log(img)

	img = np.asarray(img, float) #npasrray converts any thing to an numpyarray of the given datatype
	img = img-np.amin(img)
	img = img / float(np.amax(img))
	
	img = img * 255
	
	cv2.imwrite(imgpath_1, img)

main()