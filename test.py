import importlib

import matplotlib.pyplot as plt
from applications import MNISTNet
import loss
importlib.reload(loss)
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
from utils.datsets import MNIST
import numpy as np

mnist = MNIST()
mnist.load()
idx = np.random.randint(mnist.num_train, size=4)
print('\nFour examples of training images:')
img = mnist.x_train[idx][:,0,:,:]

# plt.figure(1, figsize=(18, 18))
# plt.subplot(1, 4, 1)
# plt.imshow(img[0])
# plt.subplot(1, 4, 2)
# plt.imshow(img[1])
# plt.subplot(1, 4, 3)
# plt.imshow(img[2])
# plt.subplot(1, 4, 4)
# plt.imshow(img[3])

model = MNISTNet()
loss = SoftmaxCrossEntropy(num_class=10)

# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 ==0:
        return lr*0.5
    else:
        return lr

adam = Adam(lr=0.001, decay=0,  sheduler_func = func)
l2 = L2(w=0.001) # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)
train_results, val_results, test_results = model.train(
    mnist,
    train_batch=30, val_batch=1000, test_batch=1000,
    epochs=2,
    val_intervals=100, test_intervals=300, print_intervals=100)