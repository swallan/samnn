import numpy as np

from CLayer import CLayer
from Layer import Layer
from Network import Network

from PIL import Image
import os
import sys

# Load in all the images

DIGITS = int(sys.argv[1]) == 1


if DIGITS:


    filePaths = ['samnn/data/0', 'samnn/data/1', 'samnn/data/2', 'samnn/data/3', 'samnn/data/4', 'samnn/data/5', 'samnn/data/6', 'samnn/data/7', 'samnn/data/8', 'samnn/data/9']#, 'samnn/data/2_','samnn/data/3_']#, 'samnn/data/4', 'samnn/data/5']
    n_classes = len(filePaths)
    inputs = []
    targets = []

    for fp in filePaths:
        for fileName in os.listdir(fp):
            try:
                inputs.append(Image.open(fp + '/' + fileName).getdata())
                # one_hot = [0, 0]
                # one_hot[int(fp.split('/')[-1])] = 1
                targets.append([int(fp.split('/')[-1].replace('_', ""))])
            except Exception as e:
                print(e)

    inputs = np.asarray(inputs) / 255
    target_temp = np.asarray(targets)
    targets = np.zeros((target_temp.size, target_temp.max()+1))
    targets[np.arange(target_temp.size),target_temp] = 1
    targets = np.zeros((target_temp.shape[0], n_classes))
    targets[np.arange(target_temp.shape[0]), target_temp.flatten()] = 1

    rng = np.random.default_rng(21321)
    shuffle_idx = list(range(len(targets)))
    rng.shuffle(shuffle_idx)


    n_sample = 1000
    inputs = inputs[shuffle_idx]#[:n_sample]
    targets = targets[shuffle_idx]#[:n_sample]
    n_features = 784
    imsize = 28, 28


else:
    # cfar
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            return dict
    n_sample = 1000
    data = unpickle('samnn/data/data_batch_1')
    targets = np.asarray(data[b'labels'] )
    inputs = data[b'data']
    inputs = np.asarray(inputs).astype(float) / 255
    target_temp = targets[:n_sample]
    targets = target_temp
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # flter = (target_temp.flatten() == 0) | (target_temp.flatten() == 1) | (target_temp.flatten() == 2) | (target_temp.flatten() == 3).flatten()
    # inputs = inputs[flter]
    # targets = targets[flter]
    # flter = (target_temp == 0) | (target_temp == 1).flatten()
    # inputs = inputs[flter]
    # targets = targets[flter]
    n_classes = max(targets) + 1
    target_temp = np.asarray(targets)

    # targets = np.zeros((target_temp.size, target_temp.max()+1))
    # targets[np.arange(target_temp.size),target_temp] = 1
    targets = np.zeros((target_temp.shape[0], n_classes))
    targets[np.arange(target_temp.shape[0]), target_temp.flatten()] = 1



    inputs = inputs[:n_sample]
    targets = targets[:n_sample]

    imsize = (32, 32, 3)

    
# shape of w is dependent on the input. 
# there are n rows for the input. 
# there are m rows for the output

# import sklearn
# from sklearn.datasets import load_iris
# from sklearn.datasets import make_blobs
# centers = [(.4, -.2), (.5, .4), (.0, .4)]
# cluster_std = [0.1, .4, .1]
# rng = np.random.default_rng(1243)
# X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=234)
# import matplotlib.pyplot as plt

# plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=100, label="Cluster1")
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=100, label="Cluster2")
# plt.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=100, label="Cluster2")


# inputs = X
# targets = np.zeros((X.shape[0], max(y) + 1))
# for i, c in enumerate(y):
#     targets[i][c] = 1

# # # target_temp = y.copy()
# # # targets = np.zeros((target_temp.shape[0], 2))
# # # targets[np.arange(target_temp.shape[0]), target_temp.flatten()] = 1
# # targets=y.reshape(100, 1)
# # inputs = X
# # print(inputs.shape, targets.shape, inputs.max())

# # number of featues (x and y coord)
# n_features = 2
# # output classes (group 1 or 2)
# n_classes = 3
# # hidden layer number of nodes
# n_hidden = 5


net = Network(
    CLayer(128, 'relu'),
    # Layer(128, 'relu'),

    Layer(n_classes, 'softmax')
)

cost, actual = net.train(inputs, targets, epochs=100, iterations=200, lr=.000005)


output_label = np.argmax(actual, axis=1)
sorted_by_type = [inputs[np.where(output_label == i)[0]] for i in range(n_classes)]
idxes = [0] * n_classes
print(f"{np.count_nonzero(output_label == np.argmax(targets, axis=1)) / len(targets)*100}% correct prediction")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))



# title = "\n".join([f"{actual[i][j]:<5.02f}" for j in range(n_classes)])
n_rows = 10
from scipy import ndimage

for i in range(n_rows * n_classes):

    plt.subplot(n_rows, n_classes, i+1) # cols, rows
    if i < n_classes:
        if DIGITS:
            plt.title(i)
        else:
            plt.title(names[i])
    what_class = i % n_classes
    idx = idxes[what_class]
    idxes[what_class] = idx + 1
    plt.axis('off')
    if len(sorted_by_type[what_class]) > idx:
        plt.tight_layout()
        if DIGITS:
            img = ndimage.rotate(sorted_by_type[what_class][idx].reshape(*imsize, order='F').T, 0)
        else:
            img = ndimage.rotate(sorted_by_type[what_class][idx].reshape(*imsize, order='F'), -90)
        plt.imshow(img)
        # title = "\n".join([f"{j}:{actual[i][j]:<5.02f}" for j in range(n_classes)])

        # plt.title(title)#f"0: {actual[i][0]:<5.02f}\n1: {actual[i][1]:<5.02f}")
    # plt.subplots_adjust(wspace=1.5)



# for i in range(25):
#     plt.subplot(5,5,i+1)    
#     plt.imshow(inputs[i].reshape(28, 28) * 255,interpolation='nearest')
#     title = "\n".join([f"{j}:{actual[i][j]:<5.02f}" for j in range(n_classes)])

#     plt.title(title)#f"0: {actual[i][0]:<5.02f}\n1: {actual[i][1]:<5.02f}")
#     plt.subplots_adjust(wspace=1.5)
print(output_label)
plt.show()
# print(np.mean((actual-targets)**2))
# # print(f"{np.count_nonzero(2 == (actual + targets))}/{actual.shape[0]}")
# # print(np.concatenate((actual, targets), axis=1))
# # import matplotlib.pyplot as plt
# actual=np.argmax(actual, axis=1)
# plt.scatter(X[actual == 0, 0], X[actual == 0, 1], color="red", s=10, label="Cluster1")
# plt.scatter(X[actual == 1, 0], X[actual == 1, 1], color="blue", s=10, label="Cluster2")
# # plt.semilogy(range(len(net.losses)), net.losses)
# plt.show()
