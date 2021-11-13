import numpy as np
import cv2
import math
from scipy.ndimage import imread
from copy import deepcopy
from math import ceil

def main():
	img = cv2.imread("image1.jpg", 0)
	depth_img = cv2.imread("image1_depth.jpg", 0)
	min_max_norm(depth_img) #The min and max pixel value in depth_img is 0 and 255. So min-max normalization will not make any difference.
	rho = 1 #camera constant
	v = 0.0001 #image distance
	u = 15 #object distance
	r = 0.2 #aperture radius

	blur_radius = r * v * abs((1/u)-1/(depth_img + 0.01)) #find the blur radius of each pixel in the image with depth map.
	sigma_image = rho * blur_radius #finding the sigma map / sigma image from the depth image.

	#Each pixel's sigma will correspond to a filter. Find the maximum filter size so that image can appropriately padded.
	max_filter_size = int(3*np.amax(sigma_image))
	if max_filter_size%2==0:
	    max_filter_size+=1
	pad = int((max_filter_size-1)/2)

	pad_image(img, pad) #pad the image
	pad_image(sigma_image, pad)

	blurred_image = convolution(pad, img, sigma_image)
	cv2.imwrite('blurred_img.jpg', blurred_image)

def convolution(pad, img, sigma_image):
	blurred_image = np.zeros(img.shape) #Create a numpy array to store the resultant blurred image.
	#i, j to loop through the img. blur_img_i and blur_img_j to loop through the blurred_image.
	blur_img_i = 0
	for i in range(pad, img.shape[0] - pad):
	    blur_img_j = 0
	    for j in range(pad, img.shape[1] - pad):	    
	        sigma_each_pixel = sigma_image[i, j]
	        kernel, kernel_size = compute_kernel(sigma_each_pixel)
	        dot_product = 0.0
	        #ki, kj to loop through the patch of the img. kernel_i, kernel_j to loop through the kernel.
	        kernel_i = 0
	        for ki in range(-(kernel_size-1)//2,(kernel_size-1)//2+1):
	            kernel_j = 0
	            for kj in range(-(kernel_size-1)//2,(kernel_size-1)//2+1):
	                product = img[i+ki,j+kj] * kernel[kernel_i,kernel_j]
	                dot_product += product
	                kernel_j += 1
	            kernel_i += 1

	        blurred_image[blur_img_i,blur_img_j] = dot_product
	        blur_img_j += 1
	    blur_img_i += 1

	return blurred_image

def compute_kernel(sigma):
    kernel_size = int(3 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = compute_gaussian_kernel(sigma, kernel_size)
    kernel = kernel/np.sum(kernel) #normalizing the kernel.
    return kernel, kernel_size

def pad_image(img, pad):
	cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

def compute_gaussian_kernel(sigma, kernel_size):
	center_i = kernel_size // 2
	center_j = kernel_size // 2
	kernel = np.zeros((kernel_size, kernel_size))
	for i in range(kernel_size):
		for j in range(kernel_size):
			euclidean_dist = math.sqrt((center_i - i) ** 2 + (center_j - j) ** 2)
			value = compute_gaussian_func_value(euclidean_dist, sigma)
			kernel[i][j] = value

	return kernel

def compute_gaussian_func_value(x, sigma):
	a = 1 / (2 * math.pi * (sigma ** 2))
	b = math.exp(- ((x ** 2) / (2 * (sigma ** 2))))
	return a * b

def min_max_norm(img):
	maximum = np.amax(img)
	minimum = np.amin(img)
	img = (img - minimum)/float(maximum - minimum)
	img = img * 255

	return img

main()