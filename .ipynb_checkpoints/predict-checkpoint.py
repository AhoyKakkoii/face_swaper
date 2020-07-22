from ISR.models import RDN
import numpy as np
from PIL import Image
import os
import scipy.misc

l = os.listdir('weights/rdn-C6-D20-G64-G064-x1')
weight_path = 'weights/rdn-C6-D20-G64-G064-x1/'+ l[-1]+'/'
l = os.listdir(weight_path)

inputf = os.listdir('result/input')

for f in inputf:
	img = Image.open('result/input/'+f)
	img =np.array(img)
	rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64,'x':1})
	for weight in l:
		if 'best' not in weight or 'rdn' not in weight:
			continue
		rdn.model.load_weights(weight_path+weight)
		sr_img=rdn.predict(img)
		scipy.misc.imsave('result/output/'+ f[:-4] + '-' + weight [:-5]+'.png', sr_img)
		print(weight, f)