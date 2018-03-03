"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np
from utils.tools import *


class Layer(object):
    """
    
    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        out_features = self.weights.shape[1]

        # Initialize outputs
        outputs = np.zeros((batch_size, out_features))

        for b in range(batch_size):
            outputs[b] = np.dot(inputs[b], self.weights) + self.bias
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        in_features = inputs.shape[1]
        out_features = in_grads.shape[1]

        # Initialize out_grads
        out_grads = np.zeros((batch_size, in_features))

        # Calculate b grads
        for o_f in range(out_features):
            self.b_grad[o_f] = np.sum(in_grads[:, o_f])

        for b in range(batch_size):
            # Calculate w_grads
            self.w_grad += np.dot(inputs[b][:, np.newaxis], in_grads[b][np.newaxis, :])

            # Calculate out_grads
            out_grads[b] = np.dot(self.weights, in_grads[b][:, np.newaxis])[:, 0]

        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/weights': self.weights,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/weights': self.w_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h']  # height of kernel
        self.kernel_w = conv_params['kernel_w']  # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros(self.out_channel)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """

        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        out_height = math.floor((in_height + self.pad * 2 - self.kernel_h) / self.stride) + 1
        out_width = math.floor((in_width + self.pad * 2 - self.kernel_w) / self.stride) + 1

        # Initialize outputs
        outputs = np.zeros((batch_size, self.out_channel, out_height, out_width))

        # Img2Col for weight
        kernel_size = self.kernel_h * self.kernel_w
        weight_trans_matrix = np.zeros((self.out_channel, self.in_channel * kernel_size))
        for out_c in range(self.out_channel):
            for in_c in range(self.in_channel):
                for h in range(self.kernel_h):
                    for w in range(self.kernel_w):
                        weight_trans_matrix[
                            out_c,
                            in_c * kernel_size + h * self.kernel_w + w] = \
                            self.weights[out_c, in_c, h, w]

        # Loop on the batch input
        for b in range(batch_size):
            padded_inputs = np.pad(inputs[b], self.pad, mode='constant')
            input_trans_matrix = np.zeros((self.in_channel * kernel_size, out_height * out_width))

            # Img2Col for inputs
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(self.in_channel):
                        for k_h in range(self.kernel_h):
                            for k_w in range(self.kernel_w):
                                input_trans_matrix[
                                    c * kernel_size + k_h * self.kernel_w + k_w,
                                    h * out_width + w] = padded_inputs[c, h * self.stride + k_h, w * self.stride + k_w]

            # Dot product for convolution
            output_trans_matrix = np.dot(weight_trans_matrix, input_trans_matrix) + self.bias[:, np.newaxis]

            # Convert back matrix
            output_matrix = np.zeros((self.out_channel, out_height, out_width))
            for c in range(self.out_channel):
                for h in range(out_height):
                    for w in range(out_width):
                        output_matrix[c, h, w] = output_trans_matrix[c, h * out_width + w]

            # Assign calculated result to outputs
            outputs[b] = output_matrix

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """

        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        out_height = in_grads.shape[2]
        out_width = in_grads.shape[3]

        # Initialize out_grads
        out_grads = np.zeros((batch_size, self.in_channel, in_height, in_width))

        # Img2Col for weight
        kernel_size = self.kernel_h * self.kernel_w
        weight_trans_matrix = np.zeros((self.out_channel, self.in_channel * kernel_size))
        for out_c in range(self.out_channel):
            for in_c in range(self.in_channel):
                for h in range(self.kernel_h):
                    for w in range(self.kernel_w):
                        weight_trans_matrix[
                            out_c,
                            in_c * kernel_size + h * self.kernel_w + w] = self.weights[out_c, in_c, h, w]

        # Calculate b grads
        for c in range(self.out_channel):
            self.b_grad[c] = np.sum(in_grads[:, c, :, :])

        for b in range(batch_size):
            padded_inputs = np.pad(inputs[b], self.pad, mode='constant')
            input_trans_matrix = np.zeros((self.in_channel * kernel_size, out_height * out_width))

            # Img2Col for inputs
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(self.in_channel):
                        for k_h in range(self.kernel_h):
                            for k_w in range(self.kernel_w):
                                input_trans_matrix[
                                    c * kernel_size + k_h * self.kernel_w + k_w,
                                    h * out_width + w] = padded_inputs[c, h * self.stride + k_h, w * self.stride + k_w]

            # Calculate w grads
            dy = np.zeros((self.out_channel, out_height * out_width))
            for c in range(self.out_channel):
                for h in range(out_height):
                    for w in range(out_width):
                        dy[c, h * out_width + w] = in_grads[b, c, h, w]
            accumulated_w_grad = np.dot(dy, input_trans_matrix.transpose())
            for out_c in range(self.out_channel):
                for in_c in range(self.in_channel):
                    for h in range(self.kernel_h):
                        for w in range(self.kernel_w):
                            self.w_grad[out_c, in_c, h, w] += \
                                accumulated_w_grad[
                                    out_c,
                                    in_c * kernel_size + h * self.kernel_w + w]

            # Calculate out grads
            d_input_trans_matrix = np.dot(weight_trans_matrix.transpose(), dy)
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(self.in_channel):
                        for k_h in range(self.kernel_h):
                            for k_w in range(self.kernel_w):
                                out_grads[b, c, h * self.stride + k_h, w * self.stride + k_w] += \
                                    d_input_trans_matrix[
                                        c * kernel_size + k_h * self.kernel_w + k_w,
                                        h * out_width + w]
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix + ':' + self.name + '/weights': self.weights,
                prefix + ':' + self.name + '/bias': self.bias
            }
            grads = {
                prefix + ':' + self.name + '/weights': self.w_grad,
                prefix + ':' + self.name + '/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >= 0) * in_grads
        out_grads = inputs_grads
        return out_grads


class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        in_channel = inputs.shape[1]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        out_height = math.floor((in_height + self.pad * 2 - self.pool_height) / self.stride) + 1
        out_width = math.floor((in_width + self.pad * 2 - self.pool_width) / self.stride) + 1

        # Initialize outputs
        outputs = np.zeros((batch_size, in_channel, out_height, out_width))

        for b in range(batch_size):
            padded_inputs = np.pad(inputs[b], self.pad, mode='constant')

            for c in range(in_channel):
                for h in range(out_height):
                    for w in range(out_width):
                        h_step = h * self.stride
                        w_step = w * self.stride
                        if self.pool_type is 'avg':
                            outputs[b, c, h, w] = np.average(padded_inputs[c,
                                                             h_step:h_step + self.pool_height,
                                                             w_step:w_step + self.pool_width])
                        elif self.pool_type is 'max':
                            outputs[b, c, h, w] = np.max(padded_inputs[c,
                                                         h_step:h_step + self.pool_height,
                                                         w_step:w_step + self.pool_width])
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        #############################################################
        # code here
        #############################################################

        # Extract necessary parameters
        batch_size = inputs.shape[0]
        in_channel = inputs.shape[1]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        out_height = in_grads.shape[2]
        out_width = in_grads.shape[3]

        # Initialize out_grads
        out_grads = np.zeros((batch_size, in_channel, in_height, in_width))

        for b in range(batch_size):
            padded_inputs = np.pad(inputs[b], self.pad, mode='constant')

            for c in range(in_channel):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_height
                        w_start = w * self.stride
                        w_end = w_start + self.pool_width
                        if self.pool_type is 'avg':
                            out_grads[b, c, h_start:h_end, w_start:w_end] += \
                                self.average_mask(self.pool_height, self.pool_width) * in_grads[b, c, h, w]
                        elif self.pool_type is 'max':
                            out_grads[b, c, h_start:h_end, w_start:w_end] += \
                                self.max_mask(padded_inputs[c, h_start:h_end, w_start:w_end]) * in_grads[b, c, h, w]
        return out_grads

    def max_mask(self, matrix):
        mask = np.zeros(matrix.shape)
        mask[np.unravel_index(np.argmax(matrix), matrix.shape)] = 1
        return mask

    def average_mask(self, height, width):
        return np.full((height, width), 1 / (height * width))


class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = inputs
        #############################################################
        # code here
        #############################################################
        if self.training:
            if self.mask is None:
                np.random.seed(self.seed)
                self.mask = np.random.binomial(1, self.ratio, size=inputs.shape)
            inputs *= self.mask
            outputs = inputs * (1 / (1 - self.ratio))
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = in_grads
        #############################################################
        # code here
        #############################################################
        if self.training:
            out_grads = in_grads * self.mask * (1 / (1 - self.ratio))
        return out_grads


class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
