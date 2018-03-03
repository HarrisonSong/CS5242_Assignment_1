import numpy as np
from layers import Convolution, Pooling, FCLayer
from loss import SoftmaxCrossEntropy
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(10, 3, 30, 30))
params = { 'kernel_h': 5,
          'kernel_w': 5,
          'pad': 0,
          'stride': 2,
          'in_channel': inputs.shape[1],
          'out_channel': 64,
}
layer = Convolution(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.Conv2D(filters=params['out_channel'],
                            kernel_size=(params['kernel_h'], params['kernel_w']),
                            strides=(params['stride'], params['stride']),
                            padding='valid',
                            data_format='channels_first',
                            input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
weights = np.transpose(layer.weights, (2, 3, 1, 0))
keras_layer.set_weights([weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('conv forward: Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))


inputs = np.random.uniform(size=(10, 3, 30, 30))
params = { 'pool_type': 'max',
           'pool_height': 5,
           'pool_width': 5,
           'pad': 0,
           'stride': 2,
}
layer = Pooling(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.MaxPooling2D(pool_size=(params['pool_height'], params['pool_width']),
                                 strides=params['stride'],
                                 padding='valid',
                                 data_format='channels_first',
                                 input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Pooling forward: Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))


inputs = np.random.uniform(size=(10, 20))
layer = FCLayer(in_features=inputs.shape[1], out_features=100)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.Dense(100, input_shape=inputs.shape[1:], use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')
# print (len(keras_layer.get_weights()))
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_layer.set_weights([layer.weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('FC forward: Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))


import warnings
warnings.filterwarnings('ignore')

batch = 10
num_class = 10
inputs = np.random.uniform(size=(batch, num_class))
targets = np.random.randint(num_class, size=batch)

loss = SoftmaxCrossEntropy(num_class)
out, _ = loss.forward(inputs, targets)

keras_inputs = K.softmax(inputs)
keras_targets = np.zeros(inputs.shape, dtype='int')
for i in range(batch):
        keras_targets[i, targets[i]] = 1
keras_out = K.mean(K.categorical_crossentropy(keras_targets, keras_inputs, from_logits=False))
print('Loss forward: Relative error (<1e-6 will be fine): ', rel_error(out, K.eval(keras_out)))