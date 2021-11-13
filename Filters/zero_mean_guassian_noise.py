import numpy as np
import cv2

def zero_mean_gauss_noise(image):
  	image_shape = image.shape
  	mean = 0
  	var = 1000
  	sigma = var ** 0.5
  	gauss_random_no = np.random.normal(mean, sigma, image_shape)
  	noise = image + gauss_random_no
  	clipped_noise = np.clip(noise, 0, 255)
  	return clipped_noise

def main():
	imgpath = "/Users/3pi/Documents/DIP/Images/Noise/Zero_mean_gaussian_noise/dragon_and_dany.jpg"
	target_imgpath = "/Users/3pi/Documents/DIP/Images/Noise/Zero_mean_gaussian_noise/dragon_and_dany_1000var_target.jpg"
	img = cv2.imread(imgpath)
	target_image = zero_mean_gauss_noise(img)
	cv2.imwrite(target_imgpath, target_image)

main()
