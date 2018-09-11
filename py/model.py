import mxnet as mx

def conv_factory(data, num_filter, num_group, kernel, prefix, num_layer, stride=(1,1), pad=(1,1)):
	conv = mx.symbol.Convolution(data=data, name='%s_conv_%s'%(prefix, num_layer), num_filter=num_filter, num_group=num_group, kernel=kernel, stride=stride, pad=pad)
	norm = mx.symbol.BatchNorm(data=conv, name='%s_norm_%s'%(prefix, num_layer))
	actv = mx.symbol.Activation(data=norm, name='%s_actv_%s'%(prefix, num_layer), act_type='relu')
	return actv

def coarse_net(data):
	conv_1 = conv_factory(data=data, num_filter=32, num_group=1, kernel=(3,3), prefix='coarse', num_layer=1)

	cvdw_2 = conv_factory(data=conv_1, num_filter=32, num_group=32, kernel=(3,3), prefix='coarse_dw', num_layer=2)
	conv_2 = conv_factory(data=cvdw_2, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=2)
	cvdw_3 = conv_factory(data=conv_2, num_filter=64, num_group=64, kernel=(3,3), prefix='coarse_dw', num_layer=3)
	conv_3 = conv_factory(data=cvdw_3, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=3)
	pool_3 = mx.symbol.Pooling(data=conv_3, name='coarse_pool_%s'%3, stride=(2,2), kernel=(2,2), pool_type='max')

	cvdw_4 = conv_factory(data=pool_3, num_filter=64, num_group=64, kernel=(3,3), prefix='coarse_dw', num_layer=4)
	conv_4 = conv_factory(data=cvdw_4, num_filter=128, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=4)
	cvdw_5 = conv_factory(data=conv_4, num_filter=128, num_group=128, kernel=(3,3), prefix='coarse_dw', num_layer=5)
	conv_5 = conv_factory(data=cvdw_5, num_filter=128, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=5)
	pool_5 = mx.symbol.Pooling(data=conv_5, name='coarse_pool_%s'%5, stride=(2,2), kernel=(2,2), pool_type='max')

	cvdw_6 = conv_factory(data=pool_5, num_filter=128, num_group=128, kernel=(3,3), prefix='coarse_dw', num_layer=6)
	conv_6 = conv_factory(data=cvdw_6, num_filter=256, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=6)
	cvdw_7 = conv_factory(data=conv_6, num_filter=256, num_group=256, kernel=(3,3), prefix='coarse_dw', num_layer=7)
	conv_7 = conv_factory(data=cvdw_7, num_filter=256, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=7)
	pool_7 = mx.symbol.Pooling(data=conv_7, name='coarse_pool_%s'%7, stride=(2,2), kernel=(2,2), pool_type='max')

	# cvdw_8 = conv_factory(data=pool_7, num_filter=256, num_group=256, kernel=(3,3), prefix='coarse_dw', num_layer=8)
	# conv_8 = conv_factory(data=cvdw_8, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=8)
	# cvdw_9 = conv_factory(data=conv_8, num_filter=512, num_group=512, kernel=(3,3), prefix='coarse_dw', num_layer=9)
	# conv_9 = conv_factory(data=cvdw_9, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=9)

	cvrd_10 = conv_factory(data=pool_7, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse_rd', num_layer=10)
	cvrd_11 = conv_factory(data=cvrd_10, num_filter=1, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse_rd', num_layer=11)
	return cvrd_11

def fine_net(data):
	conv_1 = conv_factory(data=data, num_filter=32, num_group=1, kernel=(3,3), prefix='fine', num_layer=1)

	cvdw_2 = conv_factory(data=conv_1, num_filter=32, num_group=32, kernel=(3,3), prefix='fine_dw', num_layer=2)
	conv_2 = conv_factory(data=cvdw_2, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=2)
	cvdw_3 = conv_factory(data=conv_2, num_filter=64, num_group=64, kernel=(3,3), prefix='fine_dw', num_layer=3)
	conv_3 = conv_factory(data=cvdw_3, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=3)
	pool_3 = mx.symbol.Pooling(data=conv_3, name='fine_pool_%s'%3, stride=(2,2), kernel=(2,2), pool_type='max')

	cvdw_4 = conv_factory(data=pool_3, num_filter=64, num_group=64, kernel=(3,3), prefix='fine_dw', num_layer=4)
	conv_4 = conv_factory(data=cvdw_4, num_filter=128, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=4)
	cvdw_5 = conv_factory(data=conv_4, num_filter=128, num_group=128, kernel=(3,3), prefix='fine_dw', num_layer=5)
	conv_5 = conv_factory(data=cvdw_5, num_filter=128, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=5)
	pool_5 = mx.symbol.Pooling(data=conv_5, name='fine_pool_%s'%5, stride=(2,2), kernel=(2,2), pool_type='max')

	cvdw_6 = conv_factory(data=pool_5, num_filter=128, num_group=128, kernel=(3,3), prefix='fine_dw', num_layer=6)
	conv_6 = conv_factory(data=cvdw_6, num_filter=256, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=6)
	cvdw_7 = conv_factory(data=conv_6, num_filter=256, num_group=256, kernel=(3,3), prefix='fine_dw', num_layer=7)
	conv_7 = conv_factory(data=cvdw_7, num_filter=256, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=7)
	pool_7 = mx.symbol.Pooling(data=conv_7, name='fine_pool_%s'%7, stride=(2,2), kernel=(2,2), pool_type='max')

	# cvdw_8 = conv_factory(data=pool_7, num_filter=256, num_group=256, kernel=(3,3), prefix='fine_dw', num_layer=8)
	# conv_8 = conv_factory(data=cvdw_8, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=8)
	# cvdw_9 = conv_factory(data=conv_8, num_filter=512, num_group=512, kernel=(3,3), prefix='fine_dw', num_layer=9)
	# conv_9 = conv_factory(data=cvdw_9, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=9)

	cvrd_10 = conv_factory(data=pool_7, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine_rd', num_layer=10)
	cvrd_11 = conv_factory(data=cvrd_10, num_filter=1, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine_rd', num_layer=11)
	return cvrd_11

def norm_coarse_net(data):
	conv_1 = conv_factory(data=data, num_filter=32, num_group=1, kernel=(3,3), prefix='coarse', num_layer=1)

	conv_2 = conv_factory(data=conv_1, num_filter=64, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=2)
	conv_3 = conv_factory(data=conv_2, num_filter=64, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=3)
	pool_3 = mx.symbol.Pooling(data=conv_3, name='coarse_pool_%s'%3, stride=(2,2), kernel=(2,2), pool_type='max')

	conv_4 = conv_factory(data=pool_3, num_filter=128, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=4)
	conv_5 = conv_factory(data=conv_4, num_filter=128, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=5)
	pool_5 = mx.symbol.Pooling(data=conv_5, name='coarse_pool_%s'%5, stride=(2,2), kernel=(2,2), pool_type='max')

	conv_6 = conv_factory(data=pool_5, num_filter=256, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=6)
	conv_7 = conv_factory(data=conv_6, num_filter=256, num_group=1, kernel=(3,3), pad=(1,1), prefix='coarse', num_layer=7)
	pool_7 = mx.symbol.Pooling(data=conv_7, name='coarse_pool_%s'%7, stride=(2,2), kernel=(2,2), pool_type='max')

	# cvdw_8 = conv_factory(data=pool_7, num_filter=256, num_group=256, kernel=(3,3), prefix='coarse_dw', num_layer=8)
	# conv_8 = conv_factory(data=cvdw_8, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=8)
	# cvdw_9 = conv_factory(data=conv_8, num_filter=512, num_group=512, kernel=(3,3), prefix='coarse_dw', num_layer=9)
	# conv_9 = conv_factory(data=cvdw_9, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse', num_layer=9)

	cvrd_10 = conv_factory(data=pool_7, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse_rd', num_layer=10)
	cvrd_11 = conv_factory(data=cvrd_10, num_filter=1, num_group=1, kernel=(1,1), pad=(0,0), prefix='coarse_rd', num_layer=11)
	return cvrd_11

def norm_fine_net(data):
	conv_1 = conv_factory(data=data, num_filter=32, num_group=1, kernel=(3,3), prefix='fine', num_layer=1)

	conv_2 = conv_factory(data=conv_1, num_filter=64, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=2)
	conv_3 = conv_factory(data=conv_2, num_filter=64, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=3)
	pool_3 = mx.symbol.Pooling(data=conv_3, name='fine_pool_%s'%3, stride=(2,2), kernel=(2,2), pool_type='max')

	conv_4 = conv_factory(data=pool_3, num_filter=128, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=4)
	conv_5 = conv_factory(data=conv_4, num_filter=128, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=5)
	pool_5 = mx.symbol.Pooling(data=conv_5, name='fine_pool_%s'%5, stride=(2,2), kernel=(2,2), pool_type='max')

	conv_6 = conv_factory(data=pool_5, num_filter=256, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=6)
	conv_7 = conv_factory(data=conv_6, num_filter=256, num_group=1, kernel=(3,3), pad=(1,1), prefix='fine', num_layer=7)
	pool_7 = mx.symbol.Pooling(data=conv_7, name='fine_pool_%s'%7, stride=(2,2), kernel=(2,2), pool_type='max')

	# cvdw_8 = conv_factory(data=pool_7, num_filter=256, num_group=256, kernel=(3,3), prefix='fine_dw', num_layer=8)
	# conv_8 = conv_factory(data=cvdw_8, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=8)
	# cvdw_9 = conv_factory(data=conv_8, num_filter=512, num_group=512, kernel=(3,3), prefix='fine_dw', num_layer=9)
	# conv_9 = conv_factory(data=cvdw_9, num_filter=512, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine', num_layer=9)

	cvrd_10 = conv_factory(data=pool_7, num_filter=64, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine_rd', num_layer=10)
	cvrd_11 = conv_factory(data=cvrd_10, num_filter=1, num_group=1, kernel=(1,1), pad=(0,0), prefix='fine_rd', num_layer=11)
	return cvrd_11

def net():
	label = mx.symbol.Variable('label')
	data_coarse = mx.symbol.Variable(name='data_coarse')
	data_fine = mx.symbol.Variable(name='data_fine')

	coarse_out = coarse_net(data_coarse)
	fine_out = fine_net(data_fine)
	out = coarse_out * fine_out
	return mx.symbol.MAERegressionOutput(data=out, label=label)

def empty_net():
	label = mx.symbol.Variable('label')
	data_coarse = mx.symbol.Variable(name='data_coarse')
	data_fine = mx.symbol.Variable(name='data_fine')
	p1 = mx.symbol.Pooling(data=data_coarse, name='1_pool_%s'%1, stride=(8,8), kernel=(8,8), pool_type='max')
	p2 = mx.symbol.Pooling(data=data_fine, name='2_pool_%s'%1, stride=(8,8), kernel=(8,8), pool_type='max')
	c1 = p1*p2
	out = conv_factory(data=c1, num_filter=1, num_group=1, kernel=(1,1), pad=(0,0), prefix='empty', num_layer=2)
	return mx.symbol.MAERegressionOutput(data=out, label=label)