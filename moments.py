from scipy import stats
import scipy
import glob 
import cv2
import pandas as pd 
import numpy as np
from skimage import feature
from skimage import data
from sklearn.cluster import KMeans
import torch
path = "C:/SIH/images/"
result = glob.glob(path+'/*.jpg')
lst = []
for j,path1 in enumerate(result):

	image = cv2.imread(path1)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

	image = image[:,:,0]
	#int(image1.shape)
	mat = feature.graycomatrix(image,[1],[45])
#print(mat)
	
	tamura = feature.graycoprops(mat,prop = 'contrast')
	print(tamura)
	tamura1 = feature.graycoprops(mat,prop = 'dissimilarity')
	tamura2 = feature.graycoprops(mat,prop = 'homogeneity')
	arr = np.concatenate((tamura,tamura1,tamura2),axis = 1)
	arr = torch.Tensor(arr)
	print(arr)
	#lst_tam = lst.append(arr.cpu().detach().numpy().reshape(-1))

#arr1 = np.array(arr)
#print(arr1)
#clust = KMeans(n_clusters = 15,random_state = 0)
#clust1 = clust.fit(arr1)
#print(clust1)
	#with open("tamura.txt",'a') as g:
		#g.write(str(arr))
		#g.write('\n')
	
	#can = cv2.Canny(image,100,200)
	#print(can)
	#with open("canny.txt",'a') as g:
		#g.write(str(can))
		#g.write('\n')
	
'''
	image = cv2.imread(path1)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	hue = image[0]
	saturation  = image[1]
	mean = scipy.mean(hue,axis = 0)
	print(mean)
	std = scipy.std(hue,axis = 0)
	#print(std)
	mean1 = scipy.mean(saturation,axis = 0)
	#print(mean1)
	std1 = scipy.std(saturation,axis = 0)
	#print(std1)
	con = np.concatenate((mean,std,mean1,std1),axis = 0)
	print(con)
	with open("chromatic.txt",'a') as g:
		g.write(str(con))
		g.write('\n')
'''
#hue = image[0]
#print(hue)
#saturation = image[1]
#print(saturation)
#mean = (hue+saturation/2)
#print(mean)
'''
	mean  = scipy.mean(image,axis=0)
	mean = mean.tolist()
	mean = np.concatenate(mean,axis= 0)

	#print(mean)

	#with open("mean"+str(j)+".txt",'a') as f:
		#f.write(str(mean))
		#f.write('\n')
	
	std = scipy.std(image,axis=0)
	std = std.tolist()
	std = np.concatenate(std,axis= 0)


	#with open("std"+str(j)+".txt",'a') as g:
		#g.write(str(std))
		#g.write('\n')
	skewness = stats.skew(image,axis=0)
	skewness = skewness.tolist()
	skewness = np.concatenate(skewness,axis= 0)

	#print(skewness)
	#fin  =str(mean) + str(std) + str(skewness)
	fin = np.concatenate((mean,std,skewness),axis = 0)
	print(fin)
	
	with open("moments.txt",'a') as f:
		#f.write(str(fin))
	
		#f.write('\n')
'''
