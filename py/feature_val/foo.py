#coding:utf-8

import mxnet as mx
import numpy as np
import logging
import os

from PIL import Image, ImageFilter, ImageEnhance
from collections import namedtuple

import dataiter as diter

ctx = mx.cpu(0)
img_w = 320
img_h = 240
logging.getLogger().setLevel(logging.DEBUG)

Batch = namedtuple('Batch', ['data'])

def conv_block(data, num_filter, name, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', dilate=(0,0)):
	conv = mx.symbol.Convolution(data=data, name=name, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=False)
	bn = mx.symbol.BatchNorm(data=conv)
	act = mx.symbol.Activation(data=bn, act_type=act_type)
	return act

def pool_block(data, stride=(2,2), kernel=(2,2), pool_type='max'):
	return mx.symbol.Pooling(data=data, stride=stride, kernel=kernel, pool_type=pool_type)

# =========================================================================================================

def model(train):
	data = mx.sym.Variable('data')
	label = mx.sym.Variable('label')
	c1 = conv_block(data, 64, 'conv1_1')
	c2 = conv_block(c1, 64, 'conv1_2')
	p2 = pool_block(c2)
	c3 = conv_block(p2, 128, 'conv2_1')
	c4 = conv_block(c3, 128, 'conv2_2')
	p4 = pool_block(c4)
	bkl = mx.symbol.BlockGrad(p4)
	c5 = conv_block(bkl, 256, 'conv3_1')
	c6 = conv_block(c5, 256, 'conv3_2')
	c7 = conv_block(c6, 256, 'conv3_3')
	p7 = pool_block(c7)
	c8 = conv_block(p7, 512, 'conv4_1')
	c9 = conv_block(c8, 512, 'conv4_2')
	c10 = conv_block(c9, 512, 'conv4_3')
	p10 = pool_block(c10)
	c11 = conv_block(p10, 512, 'conv5_1')
	c12 = conv_block(c11, 512, 'conv5_2')
	c13 = conv_block(c12, 512, 'conv5_3')
	co = conv_block(c13, 1, 'conv6_1',  kernel=(1,1), stride=(1,1), pad=(0,0))
	if not train:
		return mx.symbol.Group([co, c13])
	loss = mx.symbol.LinearRegressionOutput(co, label)
	return loss

def tune():
	symbol = model(True)
	arg_names = symbol.list_arguments()
	ttt_shapes, _, _ = symbol.infer_shape(data = (diter.batch_size, 3, 320, 240), label=(diter.batch_size, 1, 20, 15))
	dataiter = diter.SaliencyIter()
	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('label',))
	mod.bind(data_shapes=dataiter.provide_data, label_shapes=dataiter.provide_label)
	mod.init_params(initializer=mx.init.Uniform(scale=.1))
	sym, arg_params, aux_params = mx.model.load_checkpoint('../params/tune', 50)
	arg_params_ = {}
	for k in arg_params:
		if k in arg_names:
			arg_params_[k] = arg_params[k]
	mod.set_params(arg_params_, aux_params, allow_missing=True)
	mod.fit(
		dataiter,
		optimizer = 'adam',
		optimizer_params = {'learning_rate':0.001},
		eval_metric = 'mae',
		batch_end_callback = mx.callback.Speedometer(diter.batch_size, 5),
		epoch_end_callback = mx.callback.do_checkpoint('../params/tune', 1),
		num_epoch = 100,
	)

def feature():
	symbol = model(False)
	arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
	# dataiter = mx.io.NDArrayIter(data=mx.nd.normal(shape=(1,3,640,480), ctx=ctx), label=mx.nd.normal(shape=(1,1,40,30), ctx=ctx), batch_size=1, shuffle=True, data_name='data', label_name='label')
	dataiter = diter.SaliencyIter()
	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',))
	mod.bind(data_shapes=dataiter.provide_data)
	sym, arg_params, aux_params = mx.model.load_checkpoint('../params/vgg16', 0)
	arg_params_, aux_params_ = {}, {}
	for k in arg_params:
		if k in arg_names:
			arg_params_[k] = arg_params[k]
	for k in aux_params:
		if k in aux_names:
			aux_params_[k] = aux_params[k]
	mod.set_params(arg_params_, aux_params_, allow_missing=True)
	fname = 'COCO_test2014_000000005572'
	data = np.array(Image.open('../data/%s.jpg'%fname)).transpose((2,0,1))
	mod.forward(Batch([mx.nd.array(np.expand_dims(data, axis=0))]))
	# mod.forward(dataiter.next())
	out = mod.get_outputs()[1][0].asnumpy()
	print out.shape
	out = 255*(out-np.amin(out))/(np.amax(out)-np.amin(out))
	for i, e in enumerate(out):
		print e.shape
		img = Image.fromarray(e.astype('uint8')).resize((640,480))
		# img = img.filter(ImageFilter.GaussianBlur(10))
		# factor = 255/float(np.amax(np.array(img)))
		# img = ImageEnhance.Brightness(img).enhance(factor).filter(ImageFilter.GaussianBlur(2))
		# img = img.filter(ImageFilter.GaussianBlur(25))
		# factor = 255/float(np.amax(np.array(img)))
		# img = ImageEnhance.Brightness(img).enhance(factor).filter(ImageFilter.GaussianBlur(2))
		img.save('../data/vgg_feature/%s.png'%i)
	print 'finished'

def predict():
	root = 'E:/Dataset/MIT300/BenchmarkIMAGES/'
	symbol = model(False)
	dataiter = diter.SaliencyIter()
	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',))
	mod.bind(data_shapes=dataiter.provide_data)
	sym, arg_params, aux_params = mx.model.load_checkpoint('../params/tune', 100)
	mod.set_params(arg_params, aux_params, allow_missing=True)
	for _, _, fs in os.walk(root):
		for f in fs:
			inp = Image.open(root+f)
			w, h = inp.size
			s_axis = min(w, h)
			s_axis = min(w, h)
			if s_axis == w:
				w1, h1 = int(w*(240/float(w))), int(h*(240/float(w)))
			else:
				w1, h1 = int(w*(240/float(h))), int(h*(240/float(h)))
			data = np.array(inp.resize((w1,h1))).transpose((2,1,0))
			mod.forward(Batch([mx.nd.array(np.expand_dims(data, axis=0))]))
			out = mod.get_outputs()[0][0][0].asnumpy().transpose()
			out = 255*(out-np.amin(out))/(np.amax(out)-np.amin(out))
			img = Image.fromarray(out.astype('uint8')).resize((w,h))
			img = img.filter(ImageFilter.GaussianBlur(10))
			factor = 255/float(np.amax(np.array(img)))
			img = ImageEnhance.Brightness(img).enhance(factor).filter(ImageFilter.GaussianBlur(2))
			img = img.filter(ImageFilter.GaussianBlur(25))
			factor = 255/float(np.amax(np.array(img)))
			img = ImageEnhance.Brightness(img).enhance(factor).filter(ImageFilter.GaussianBlur(2))
			img.save('../data/mit300/%s.png'%(f.split('.')[0]))
			print '%s finished'%f

if __name__ == '__main__':
	feature()