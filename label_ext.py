import torch 
import torch.nn as nn
from torchvision import models
from torchsummary import summary 
import torchvision.transforms as transforms
import glob 
from sklearn.cluster import KMeans
import cv2
import numpy as np
import pandas as pd 

path = "/content/drive/MyDrive/SIH/Images2/"
#$weights =  models.ResNet50_Weights.DEFAULT
model = models.vgg16(pretrained = True)

model = model.type(torch.cuda.FloatTensor)
print(summary(model,(3,256,256)))

class extract(nn.Module):
	def __init__(self,model):
		super(extract,self).__init__()
		self.features  = list(model.features)
		self.features = nn.Sequential(*self.features)
		self.pooling = model.avgpool
		self.flatten = nn.Flatten()
		#self.linear = nn.Linear(1,10)
		self.fc = model.classifier[1]

	def forward(self,input):
		out = self.features(input)
		out = self.pooling(out)
		out = self.flatten(out)
		out = self.fc(out)
		return out 
#model = models.resnet50(weights = weights)

updated = extract(model)
print(summary(updated,(3,256,256)))

transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
features = []
result  = glob.glob(path+"/*.jpg")
for i,path1 in enumerate(result):

	image = cv2.imread(path1)
	image = transform(image)
	image = image.type(torch.cuda.FloatTensor)
#image = image.to(device)
	#print(type(image))
#feature = updated(image)
	with torch.no_grad():
		feature = updated(image)
	features.append(feature.cpu().detach().numpy().reshape(-1))
features = np.array(features)
with open("/content/drive/MyDrive/SIH/features"+str(i)+".txt",'a') as f:

    f.write(str(features))
    f.write("\n")

print(features)

#features = features.reshape(1,-1)
clust = KMeans(n_clusters = 1,random_state= 0)
#features  =features.cpu()
clust1 = clust.fit(features)
labels = clust1.labels_
number = clust1.n_features_in_
name = clust1.feature_names_in_
print(clust1)
print(number)
print(name)
#labels.to_csv("labels.csv",index = False)
#np.savetxt("/content/drive/MyDrive/SIH/labels.txt",labels)






