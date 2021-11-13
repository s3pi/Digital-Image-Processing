import numpy as np
import cv2
import math

img = cv2.imread("/Users/3pi/Documents/DIP/Images/Noise/emma_50*50.jpg")
img = img/255.
kernel_size = 5
sigma = 3

def compute_gaussian_func_value(x, sigma):
	a = 1 / (2 * math.pi * (sigma ** 2))
	b = math.exp(- ((x ** 2) / (2 * (sigma ** 2))))
	return a * b
rows, cols, colors = img.shape

convolved_image = np.zeros((img.shape[0]-int(2*math.floor(kernel_size/2)),img.shape[1]-int(2*math.floor(kernel_size/2)),3))
i = 0
for h in range(int(math.floor(kernel_size/2)),img.shape[0]-int(math.floor(kernel_size/2))):
	j=0
	for k in range(int(math.floor(kernel_size/2)),img.shape[1]-int(math.floor(kernel_size/2))):
		vec10=np.zeros((kernel_size*kernel_size,))
		vec11=np.zeros((kernel_size*kernel_size,))
		vec12=np.zeros((kernel_size*kernel_size,))	
		count = 0
		for k1 in range(-int(math.floor(kernel_size/2)),int(math.floor(kernel_size/2))):
			for k2 in range(-int(math.floor(kernel_size/2)),int(math.floor(kernel_size/2))):
				vec10[count] = img[h+k1][k+k2][0]
				vec11[count] = img[h+k1][k+k2][1]
				vec12[count] = img[h+k1][k+k2][2]
				count+=1
		sum0=0
		sum1=0
		sum2=0
		sumk0=0
		sumk1=1
		sumk2=2
		for m in range(int(math.floor(kernel_size/2)),img.shape[0]-int(math.floor(kernel_size/2))):
			for n in range(int(math.floor(kernel_size/2)),img.shape[1]-int(math.floor(kernel_size/2))):
				vec20=np.zeros((kernel_size*kernel_size,))
				vec21=np.zeros((kernel_size*kernel_size,))
				vec22=np.zeros((kernel_size*kernel_size,))
				count = 0
				for k1 in range(-int(math.floor(kernel_size/2)),int(math.floor(kernel_size/2))):
					for k2 in range(-int(math.floor(kernel_size/2)),int(math.floor(kernel_size/2))):
						vec20[count] = img[m+k1][n+k2][0]
						vec21[count] = img[m+k1][n+k2][1]
						vec22[count] = img[m+k1][n+k2][2]
						count+=1
				dist0 = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec10, vec20)]))
				dist1 = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec11, vec21)]))
				dist2 = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec12, vec22)]))
				weight0 = gauss_output(dist0,sigma)
				weight1 = gauss_output(dist1,sigma)
				weight2 = gauss_output(dist2,sigma)
				sumk0+=weight0
				sumk1+=weight1
				sumk2+=weight2
				sum0+=weight0*img[m,n,0]
				sum1+=weight1*img[m,n,1]
				sum2+=weight2*img[m,n,2]
		sum0=sum0/sumk0
		sum1=sum1/sumk1
		sum2=sum2/sumk2
		convolved_image[i,j,0] = sum0
		convolved_image[i,j,1] = sum1
		convolved_image[i,j,2] = sum2
		j+=1
	i+=1

minimum = np.amin(convolved_image)
maximum = np.amax(convolved_image)

convolved_image = (convolved_image - minimum)/float(maximum - minimum)
convolved_image = convolved_image * 255
cv2.imwrite("/Users/3pi/Documents/DIP/Images/Noise/emma_50*50_nlm.jpg",convolved_image)