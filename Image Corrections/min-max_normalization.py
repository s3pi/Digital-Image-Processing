sdf;lkjimport cv2
import numpy as np
 
def main():
	print(cv2.__version__)
	imgpath = "/Users/3pi/Documents/DIP/Current/sample_images/kfp.jpg"
	imgpath_1 = "/Users/3pi/Documents/DIP/Current/sample_images/kfp_min-max.jpg"
	img = cv2.imread(imgpath) #reading the image from the path and storing it as an array
	print(img.shape) #prints the 3 tuple - (height, depth, color - 3 for RBG, 2 - for gray, 3 component is 0)

	
	minima = np.amin(img) #min intensity of the given image
	maxima = np.amax(img)
	print(minima)
	print(maxima)

	img = img-np.amin(img) #every pixel is decreased by a min value. Beauty of numpy.  
	minima_1 = np.amin(img)
	maxima_1 = np.amax(img)
	print(minima_1)
	print(maxima_1)

	img = img / float(np.amax(img))
	img = img * 255
	cv2.imwrite(imgpath_1, img)

'''if the min and max were 0 and 255, min-max normalization will not make any difference.
'''


main()

'''
if name == "__main__":
	main()
'''