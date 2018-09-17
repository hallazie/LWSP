import mxnet as mx
import numpy as np
import logging
import time
import traceback

print mx.__version__

logging.getLogger().setLevel(logging.DEBUG)
ctx = mx.cpu(0)
batch_size = 1
sample_size = 10

def conv_factory(data, num_filter, kernel, prefix, num_layer, num_group=1, stride=(1,1), pad=(1,1)):
	conv = mx.symbol.Convolution(data=data, name='%s_conv_%s'%(prefix, num_layer), num_filter=num_filter, num_group=num_group, kernel=kernel, stride=stride, pad=pad)
	norm = mx.symbol.BatchNorm(data=conv, name='%s_norm_%s'%(prefix, num_layer))
	actv = mx.symbol.Activation(data=norm, name='%s_actv_%s'%(prefix, num_layer), act_type='relu')
	return actv

def pool_factory(data, kernel, pool_type, num_layer, stride=(2,2), pad=(0,0)):
	return mx.symbol.Pooling(data=data, name='pool_%s'%num_layer, stride=(2,2), kernel=kernel, pad=pad, pool_type=pool_type)

def deepgaze2():
	data = mx.symbol.Variable('data')
	label = mx.symbol.Variable('label')
	pf = 'deepgaze'
	# 640*480
	c1 = conv_factory(data, 64, (3,3), pf, 1)
	c2 = conv_factory(c1, 64, (3,3), pf, 2)
	p2 = pool_factory(c2, (2,2), 'max', 2)

	c3 = conv_factory(p2, 128, (3,3), pf, 3)
	c4 = conv_factory(c3, 128, (3,3), pf, 4)
	p4 = pool_factory(c4, (2,2), 'max', 4)

	c5 = conv_factory(p4, 256, (3,3), pf, 5)
	c6 = conv_factory(c5, 256, (3,3), pf, 6)
	c7 = conv_factory(c6, 256, (3,3), pf, 7)
	c8 = conv_factory(c7, 256, (3,3), pf, 8)
	p8 = pool_factory(c8, (2,2), 'max', 8)

	c9 = conv_factory(p8, 512, (3,3), pf, 9)
	c10 = conv_factory(c9, 512, (3,3), pf, 10)
	c11 = conv_factory(c10, 512, (3,3), pf, 11)
	c12 = conv_factory(c11, 512, (3,3), pf, 12)
	p12 = pool_factory(c12, (2,2), 'max', 12)

	c13 = mx.symbol.Convolution(data=p12, name='deepgaze_conv_13', num_filter=512, kernel=(3,3), stride=(1,1), pad=(1,1))
	n13 = mx.symbol.BatchNorm(data=c13, name='deepgaze_norm_13')
	a13 = mx.symbol.Activation(data=n13, name='deepgaze_actv_13', act_type='relu')	

	c14 = conv_factory(a13, 512, (3,3), pf, 14)
	c15 = conv_factory(c14, 512, (3,3), pf, 15)
	c16 = conv_factory(c15, 512, (3,3), pf, 16)

	u1 = mx.symbol.UpSampling(n13, scale=8, sample_type='nearest')
	u2 = mx.symbol.UpSampling(a13, scale=8, sample_type='nearest')
	u3 = mx.symbol.UpSampling(c14, scale=8, sample_type='nearest')
	u4 = mx.symbol.UpSampling(c15, scale=8, sample_type='nearest')
	u5 = mx.symbol.UpSampling(c16, scale=8, sample_type='nearest')

	concat = mx.symbol.concat(u1,u2,u3,u4,u5)

	r1 = conv_factory(concat, 16, (1,1), 'readout', 1, pad=(0,0))
	r2 = conv_factory(r1, 32, (1,1), 'readout', 2, pad=(0,0))
	r3 = conv_factory(r2, 2, (1,1), 'readout', 3, pad=(0,0))
	r4 = conv_factory(r3, 1, (1,1), 'readout', 4, pad=(0,0))

	return mx.symbol.LinearRegressionOutput(data=r4, label=label)

def dataiter():
	data = {'data':np.ones(shape=(50,3,320,240))}
	label = {'label':np.ones(shape=(50,1,160,120))}
	diter = mx.io.NDArrayIter(data, label, batch_size, True, last_batch_handle='discard')
	return diter

def run():

	diter = dataiter()
	symbol = deepgaze2()
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
		epoch_end_callback = mx.callback.do_checkpoint('../params/deepgaze2', 10),
		num_epoch = 100,
		)

def test():
	# diter = dataiter()
	symbol = deepgaze2()
	_, arg_params, aux_params = mx.model.load_checkpoint('../params/deepgaze2', 20)
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