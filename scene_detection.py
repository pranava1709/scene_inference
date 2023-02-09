from scipy import stats
import scipy
import glob 
import cv2
import pandas as pd 
import numpy as np
from skimage import feature
from skimage import filters
from skimage import data
import sklearn
from sklearn.cluster import KMeans
feat = []
features = []
path = "C:/SIH/train/"
result = glob.glob(path+'/*.jpg')
for j,path1 in enumerate(result):
	image = cv2.imread(path1)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	#image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	
	
	mean  = scipy.mean(image,axis=0)
	

	
	mean = mean.tolist()
	std = scipy.std(image,axis=0)
	std = std.tolist()
	

	skewness = stats.skew(image,axis=0)
	skewness = skewness.tolist()
	skewness = np.concatenate((mean,std,skewness),axis= 1)
	norm_mom = np.linalg.norm(skewness)
	features.append(norm_mom)





	hue = image[0]
	saturation  = image[1]
	meana = scipy.mean(hue,axis = 0)
	

	stda= scipy.std(hue,axis = 0)
	#print(std)
	

	mean1 = scipy.mean(saturation,axis = 0)
	#print(mean1)
	

	std1 = scipy.std(saturation,axis = 0)
	#print(std1)


	con = np.concatenate((meana,stda,mean1,std1),axis = 0)
	#print(con)
	norm_con = np.linalg.norm(con)
	features.append(norm_con)
	image  = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
	can = cv2.Canny(image,100,200)
	can = np.array(can)
	norm_can = np.linalg.norm(can)

	features.append(norm_can)

	gabor  = filters.frangi(image)
	image = image[:,:,:]
	print(gabor.shape)
	df = pd.DataFrame()
	num = 1
  
	for i in range(1,6):
		i = (i/4)*3.14
		for h in range(3,9):
			for j in range(1,4):
				for k in range(1,2):

					kernel = cv2.getGaborKernel((39,39),h,j,i,k,0,ktype=cv2.CV_32F)
					img  = cv2.filter2D(image,cv2.CV_8UC3,kernel)
					img = img.reshape(-1)
			#	df[Gabor] = img
			#	df.to_csv("Gabor.csv")
					num = num + 1
	print(img)
	gab = np.array(img)
	gab_norm = np.linalg.norm(gab)
	features.append(gab_norm)
  
	
	image = image[:,:,0]
	mat = feature.graycomatrix(image,[1],[45])
	tamura = feature.graycoprops(mat,prop = 'contrast')
	tamura1 = feature.graycoprops(mat,prop = 'dissimilarity')
	tamura2 = feature.graycoprops(mat,prop = 'homogeneity')
	#with open("chromatic.txt",'a') as g:
		#g.write(str(con))
		#g.write('\n')


	arr1= np.concatenate((tamura,tamura1,tamura2),axis = 1)
	#print(arr.shape)
	#arr= arr.reshape(arr[0],arr[1]*arr[2])

	#arr = arr.flat()
	#print(dir(arr))
	#arr = features.tolist()
	norm_tam = np.linalg.norm(arr1)
	#print(norm)


	
	#print(arr)
	
	#print(arr.size)
	
	#print(arr)
	features.append(norm_tam)
#print(feat)
features  = np.array(features)
features = features.reshape(-1,1)

#features = features.tolist()
print(features)
clust = KMeans(n_clusters = 6,random_state =0)
clust1 = clust.fit(features)
centers = clust1.cluster_centers_
print(centers)
with open("centers.txt",'a') as f:
	f.write(str(centers))
	f.write('\n')