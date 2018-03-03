from layers import Convolution, Dropout, Pooling, FCLayer
from loss import SoftmaxCrossEntropy
import numpy as np
from utils.check_grads import check_grads_layer, check_grads_loss

## Conv
batch = 10
conv_params={
    'kernel_h': 3,
    'kernel_w': 3,
    'pad': 0,
    'stride': 2,
    'in_channel': 3,
    'out_channel': 10
}
in_height = 10
in_width = 20
out_height = 1+(in_height+2*conv_params['pad']-conv_params['kernel_h'])//conv_params['stride']
out_width = 1+(in_width+2*conv_params['pad']-conv_params['kernel_w'])//conv_params['stride']
inputs = np.random.uniform(size=(batch, conv_params['in_channel'], in_height, in_width))
in_grads = np.random.uniform(size=(batch, conv_params['out_channel'], out_height, out_width))
conv = Convolution(conv_params)
check_grads_layer(conv, inputs, in_grads)

## Dropout
ratio = 0.1
height = 10
width = 20
channel = 10
np.random.seed(1234)
inputs = np.random.uniform(size=(batch, channel, height, width))
in_grads = np.random.uniform(size=(batch, channel, height, width))
dropout = Dropout(ratio, seed=1234)
dropout.set_mode(True)
check_grads_layer(dropout, inputs, in_grads)

## Pooling
params = { 'pool_type': 'max',
           'pool_height': 5,
           'pool_width': 5,
           'pad': 0,
           'stride': 2,
}
batch = 10
channel = 10
in_height = 10
in_width = 20
out_height = 1+(in_height+2*params['pad']-params['pool_height'])//params['stride']
out_width = 1+(in_width+2*params['pad']-params['pool_width'])//params['stride']
layer = Pooling(params)
inputs = np.random.uniform(size=(batch, channel, in_height, in_width))
in_grads = np.random.uniform(size=(batch, channel, out_height, out_width))
check_grads_layer(layer, inputs, in_grads)

## FC
batch = 10
in_features = 20
out_features = 100
inputs = np.random.uniform(size=(batch, in_features))
in_grads = np.random.uniform(size=(batch, out_features))
layer = FCLayer(in_features=inputs.shape[1], out_features=100)
check_grads_layer(layer, inputs, in_grads)

## Loss
batch = 10
num_class = 10
inputs = np.random.uniform(size=(batch, num_class))
targets = np.random.randint(num_class, size=batch)

loss = SoftmaxCrossEntropy(num_class)
check_grads_loss(loss, inputs, targets)
