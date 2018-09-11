#coding:utf-8

import mxnet as mx
import numpy as np
import model_v2 as model
import traceback
import logging
import old_model
import time

from PIL import Image
from config import *

print mx.__version__

from collections import namedtuple
Batch = namedtuple('Batch', ['data_coarse','data_fine'])

logging.getLogger().setLevel(logging.DEBUG)

ctx = mx.gpu(0)
batch_size = 48
sample_size = 5

def dataiter():
	data = {'data_coarse':np.random.uniform(0,255,(batch_size,3,80,80)), 'data_fine':np.random.uniform(0,255,(batch_size,3,80,80))}
	label = {'label':np.ones(shape=(batch_size,1,10,10))}
	diter = mx.io.NDArrayIter(data, label, batch_size, True, last_batch_handle='discard')
	return diter

def train():

	diter = dataiter()
	symbol = model.net()
	arg_names = symbol.list_arguments()

	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data_coarse','data_fine'), label_names=('label',))
	mod.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)

	mod.init_params(initializer=mx.init.Uniform(scale=.1))

	mod.fit(
		diter,
		optimizer = 'sgd',
		optimizer_params = {'learning_rate':0.01},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(batch_size, 1),
		epoch_end_callback = mx.callback.do_checkpoint(MODEL_PREFIX, 1),
		num_epoch = 10,
		)
	
def test():
	# diter = dataiter()
	symbol = model.net()
	# symbol = old_model.empty_net()

	_, arg_params, aux_params = mx.model.load_checkpoint('../params/lwsp', 1)
	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data_coarse','data_fine'), label_names=None)
	mod.bind(for_training=False, data_shapes=[('data_coarse', (batch_size,3,80,80)),('data_fine', (batch_size,3,80,80))], label_shapes=mod._label_shapes)
	mod.set_params(arg_params, aux_params, allow_missing=True)

	print time.asctime(time.localtime(time.time()))
	start = time.time()

	for k in range(5):
		for i in range(sample_size):
			try:
				batch = mx.io.DataBatch([mx.nd.random.uniform(0,255,(batch_size,3,80,80)), mx.nd.random.uniform(0,255,(batch_size,3,80,80))])
				mod.forward(batch)
				saliency = mod.get_outputs()[0].asnumpy()
			except:
				traceback.print_exc()
				return
		print '%s: %s'%(k,time.asctime(time.localtime(time.time())))

	print time.asctime(time.localtime(time.time()))
	end = time.time()
	print '%s seconds per sample'%str(abs(end-start)/float(sample_size*5))

if __name__ == '__main__':
	train()