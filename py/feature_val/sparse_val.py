#coding:utf-8

# import os
# import numpy as np

# from PIL import Image

# vgg_path = '../data/vgg_feature/'
# sal_path = '../data/saliency_512/'

# def val():
# 	vgg_avg = avg(vgg_path)
# 	sal_avg = avg(sal_path)
# 	print 'vgg:\t%s'%(vgg_avg.sum()/float(640*480*512))
# 	print 'sal:\t%s'%(sal_avg.sum()/float(640*480*512))

# def avg(path):
# 	ret = np.zeros((480,640))
# 	for _,_,fs in os.walk(path):
# 		for f in fs:
# 			ret += np.array(Image.open(path+f).convert('L'))
# 	return ret

import cPickle as pkl
import numpy as np
import keras

def func():
	with open('../params/mlnet_salicon_weights.pkl', 'r') as f:
		mlw = pkl.load(f)  
		print mlw

if __name__ == '__main__':
	func()