import skimage
from skimage import filters
import cv2
import pandas as pd
from skimage import feature
import numpy as np
image = cv2.imread("1.jpg")

gabor  = filters.frangi(image)
print(gabor.shape)
df = pd.DataFrame()
num = 1
'''
for i in range(1,6):
	i = (i/4)*3.14
	for h in range(3,9):
		for j in range(1,4):
			for k in range(1,2):
				Gabor = "gabor" + str(num)

				kernel = cv2.getGaborKernel((39,39),h,j,i,k,0,ktype=cv2.CV_32F)
				img  = cv2.filter2D(image,cv2.CV_8UC3,kernel)
				img = img.reshape(-1)
				#df[Gabor] = img
				#df.to_csv("Gabor.csv")
				num = num + 1
				print(img)
'''
'''
file = pd.read_csv("Gabor.csv")
print(file.shape)
file1 = np.array(file)
print(file1.shape)
with open("gabor.txt",'a') as h:
	h.write(str(file1))
'''
#image = np.array(image)
#image   = np.squeeze(image,axis = 2)
#can = feature.canny(image)
#print(can)
print(image.shape)
image = image[:,:,0]
print(image.shape)


mat = feature.graycomatrix(image,[1],[45])
#print(mat)
tamura = feature.graycoprops(mat,prop = 'contrast')
tamura1 = feature.graycoprops(mat,prop = 'dissimilarity')
tamura2 = feature.graycoprops(mat,prop = 'homogeneity')
arr = np.concatenate((tamura,tamura1,tamura2),axis = 1)
print(arr)