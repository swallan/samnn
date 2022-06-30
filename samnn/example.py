import numpy as np

from Layer import Layer
from Network import Network

inputs = np.asarray([[0.1, .7],])

# shape of w is dependent on the input. 
# there are n rows for the input. 
# there are m rows for the output



targets = np.asarray([[1, 0],])

print(inputs.shape, targets.shape)
net = Network(
    Layer(3, 'relu'),
    Layer(2, 'relu')

)

print(net.train(inputs, targets, iterations=2).shape)

