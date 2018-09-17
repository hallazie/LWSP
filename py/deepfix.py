import mxnet as mx
import numpy as np
import logging
import time
import traceback

print mx.__version__

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

logging.getLogger().setLevel(logging.DEBUG)
ctx = mx.cpu(0)
batch_size = 1
sample_size = 10

def conv_factory(data, num_filter, kernel, prefix, num_layer, num_group=1, stride=(1,1), pad=(1,1)):
	conv = mx.symbol.Convolution(data=data, name='%s_conv_%s'%(prefix, num_layer), num_filter=num_filter, num_group=num_group, kernel=kernel, stride=stride, pad=pad)
	norm = mx.symbol.BatchNorm(data=conv, name='%s_norm_%s'%(prefix, num_layer))
	actv = mx.symbol.Activation(data=norm, name='%s_actv_%s'%(prefix, num_layer), act_type='relu')
	return actv

def pool_factory(data, num_layer, stride, kernel, pool_type, pad=(0,0)):
	return mx.symbol.Pooling(data=data, name='pool_%s'%num_layer, stride=(2,2), kernel=(2,2), pad=pad, pool_type=pool_type)

def inception_factory(data, num_layer):
	c1_1 = conv_factory(data=data, num_filter=128, kernel=(1,1), stride=(1,1), pad=(0,0), prefix='deepfix_1', num_layer=num_layer)

	c2_1 = conv_factory(data=data, num_filter=128, kernel=(1,1), stride=(1,1), pad=(0,0), prefix='deepfix_2_1', num_layer=num_layer)
	c2_2 = conv_factory(data=c2_1, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), prefix='deepfix_2_2', num_layer=num_layer)

	c3_1 = conv_factory(data=data, num_filter=32, kernel=(1,1), stride=(1,1), pad=(0,0), prefix='deepfix_3_1', num_layer=num_layer)
	c3_2 = conv_factory(data=c3_1, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), prefix='deepfix_3_2', num_layer=num_layer)

	# p4_1 = pool_factory(data=data, num_layer=num_layer, stride=(1,1), kernel=(3,3), pad=(1,1), pool_type='max')
	c4_2 = conv_factory(data=data, num_filter=64, kernel=(1,1), stride=(1,1), pad=(0,0), prefix='deepfix_4_2', num_layer=num_layer)
	return mx.symbol.concat(c1_1,c2_2,c3_2,c4_2)

def deepfix():
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('label')
	# 640*480
	c1 = conv_factory(data=data, num_filter=64, kernel=(3,3), prefix='deepfix', num_layer=1)
	c2 = conv_factory(data=c1, num_filter=64, kernel=(3,3), prefix='deepfix', num_layer=2)
	p2 = pool_factory(data=c2, num_layer=2, stride=(2,2), kernel=(3,3), pad=(1,1), pool_type='max')
	# 320*240
	c3 = conv_factory(data=p2, num_filter=128, kernel=(3,3), prefix='deepfix', num_layer=3)
	c4 = conv_factory(data=c3, num_filter=128, kernel=(3,3), prefix='deepfix', num_layer=4)
	p4 = pool_factory(data=c4, num_layer=4, stride=(2,2), kernel=(3,3), pad=(1,1), pool_type='max')
	# 160*120
	c5 = conv_factory(data=p4, num_filter=256, kernel=(3,3), prefix='deepfix', num_layer=5)
	c6 = conv_factory(data=c5, num_filter=256, kernel=(3,3), prefix='deepfix', num_layer=6)
	c7 = conv_factory(data=c6, num_filter=256, kernel=(3,3), prefix='deepfix', num_layer=7)
	p7 = pool_factory(data=c7, num_layer=7, stride=(2,2), kernel=(3,3), pad=(1,1), pool_type='max')
	# 80*60
	c8 = conv_factory(data=p7, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=8)
	c9 = conv_factory(data=c8, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=9)
	c10 = conv_factory(data=c9, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=10)
	p10 = pool_factory(data=c10, num_layer=10, stride=(1,1), kernel=(3,3), pad=(0,0), pool_type='max')
	# 40*30
	c11 = conv_factory(data=p10, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=11)
	c12 = conv_factory(data=c11, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=12)
	c13 = conv_factory(data=c12, num_filter=512, kernel=(3,3), prefix='deepfix', num_layer=13)
	# 40*30
	i14 = inception_factory(data=c13, num_layer=14)
	i15 = inception_factory(data=i14, num_layer=15)

	c16 = conv_factory(data=i15, num_filter=512, kernel=(5,5), pad=(2,2), prefix='deepfix', num_layer=16)
	c17 = conv_factory(data=c16, num_filter=512, kernel=(5,5), pad=(2,2), prefix='deepfix', num_layer=17)

	c18 = conv_factory(data=c17, num_filter=1, kernel=(1,1), pad=(0,0), prefix='deepfix', num_layer=18)

	return mx.symbol.LinearRegressionOutput(data=c18, label=label)

def dataiter():
	data = {'data':np.ones(shape=(sample_size,3,640,480))}
	label = {'label':np.ones(shape=(sample_size,1,40,30))}
	diter = mx.io.NDArrayIter(data, label, batch_size, True, last_batch_handle='discard')
	return diter

def run():

	diter = dataiter()
	symbol = deepfix()
	arg_names = symbol.list_arguments()

	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=('label',))
	mod.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)

	mod.init_params(initializer=mx.init.Uniform(scale=.1))

	mod.fit(
		diter,
		optimizer = 'sgd',
		optimizer_params = {'learning_rate':0.001},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(batch_size, 1),
		epoch_end_callback = mx.callback.do_checkpoint('../params/deepfix', 10),
		num_epoch = 100,
		)

def test():
	# diter = dataiter()
	symbol = deepfix()
	_, arg_params, aux_params = mx.model.load_checkpoint('../params/deepfix', 10)
	mod = mx.mod.Module(symbol=symbol, context=ctx, data_names=('data',), label_names=None)
	mod.bind(for_training=False, data_shapes=[('data', (1,3,640,480))], label_shapes=mod._label_shapes)
	mod.set_params(arg_params, aux_params, allow_missing=True)

	print time.asctime(time.localtime(time.time()))
	start = time.time()

	for k in range(10):
		for i in range(sample_size):
			try:
				batch = mx.io.DataBatch([mx.nd.random.uniform(0,255,(1,3,640,480))])
				mod.forward(batch)
				saliency = mod.get_outputs()[0].asnumpy()
			except:
				# diter.reset()
				traceback.print_exc()
		print '%s: %s'%(k,time.asctime(time.localtime(time.time())))

	print time.asctime(time.localtime(time.time()))
	end = time.time()
	print '%s seconds per sample'%str(abs(end-start)/float(sample_size*10))
	
if __name__ == '__main__':
	test()