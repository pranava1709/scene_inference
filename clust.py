import numpy as np
from sklearn.cluster import KMeans
import ast 
import re
import pandas as pd 
lst_fr = []
with open("tamura.txt",'r') as f:
	for i in f:
		rowy = np.asarray(i)



		#files = np.ar(i)
		#files = i.tolist()
		
		lst_fr.append(i)
		b = rowy.tolist()


		b= b.replace('[','')
		b= b.replace(']','')
		b= np.asarray(b)
		#print(type(b))


		print(b.shape)		


		#with open("tamura2.txt",'a') as g:
		#	g.write('[')
		#	g.write(str(b))
		#	g.write(']')
			
		#	g.write('\n')


#with open("tamura2.txt",'r') as h:
#	file = h.readlines()
#	print(type(file))
#	lst2 = [ast.literal_eval(q) for q in file]
#	print(lst2)
'''

		#print(b)

		#res = np.array(re.split("/s+", b.replace('[','').replace(']','')), dtype=float)
	#print(lst_fr.tolist())
	#lst_fr1 = [a.strip() for a in lst_fr]
	#lst_fr2 = [a.tolist() for a in lst_fr]  

	#lst_fr2 = [a.replace("'"," ") for a in lst_fr1]
#print(lst_fr)
#print(lst_fr2)
#lst2 = [ast.literal_eval(q) for q in lst_fr]
#print(lst2)


'''
'''
joined_string = ",". join(a for a in lst_fr2 if a not in "," )
print(joined_string)

#y = "".join(a for a in joined_string if a not in "'")
#for j in joined_string:
	#print(j)
#print(y)
#print(arr)
#result = np.array(re.split("s+", y.replace('[','').replace(']','')), dtype = float)
#y1 ="".join(b for b in result if b not in "'")

#
#print(joined_string)
#arr = np.char.replace(arr,"'","")
#print(arr)
clust = KMeans(n_clusters = 15, random_state =0)
#arr = np.array(result)
#print(arr)
clust1 = clust.fit(joined_string)
#print(clust1)

'''

	
