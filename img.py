from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import math


path =  None #'C:/Users/Yu/Desktop/bnl2019summer/project/galaxy/galaxy_zoo'
out_path = 'galaxy_zoo/'

def extract(array, folder, part=None):
	train = folder[:-4]
	valid = train[:-5]+'valid'
	t = 'single' if 'individuals' in folder else 'blend'
	for i in range(len(array)):
		n = i+1
		f = train
		#plt.imshow(array[i], cmap='gray')

		if i > 8000:
			n = i-8000
			f = valid
		nzero = 4-math.floor(math.log(n, 10))
		nzero = nzero-1 if n==1000 else nzero
		img_name = t + '_' + f.split('_')[1][0]+'_'+'0'*nzero+str(n)
		img_name = img_name + '_' + str(part) if part else img_name
		
		if img_name+'.png' in os.listdir(out_path + f):
			print(img_name+'.png', 'exsists')
			continue

		img_name = out_path + f + '/' + img_name + '.png'
		matplotlib.image.imsave(img_name, array[i])
		print(img_name)

		img = Image.open(img_name)
		if len(img.split())==3:
			continue

		img = img.convert('RGB')
		img.save(img_name)
		print(img_name, 'converted --> channels#:', len(img.split()))





if path == None:
	path = input('The path to the folder containing images in npy format:')

#create folders
for f in os.listdir(path):
	if 'npy' not in f:
		continue

	out = out_path
	if 'test' in f:
		out = out_path+f[:-8]+'valid'

	else:
		out = out_path+f[:-4]
	if out[11:] in os.listdir(out_path[:-1]):
		print(out, 'exists')
		continue

	os.makedirs(out)
	print(out, 'created')

for f in os.listdir(path):
	if 'npy' not in f or 'train' not in f:
		continue

	out = out_path+f[:-4]
	img_arrays = np.load(path+'/'+f)
	if len(img_arrays)<10:
		i=1
		for arr in img_arrays:
			extract(arr, f, part=i)
			i+=1
	else:
		extract(arr, f)

os.makedirs('model/21')
os.makedirs('model/24')
os.makedirs('result/input')
os.makedirs('result/output')
os.makedirs('weights')
os.makedirs('logs')
