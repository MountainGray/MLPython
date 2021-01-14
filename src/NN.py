import numpy as np
from numpy import random

train_labels="data/train-labels-idx1-ubyte"
train_images="data/train-images-idx3-ubyte"

class Network(object):
    def __init__(self, sizes) -> None:
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases = [np.random(i,1) for i in sizes[1:]]
        self.weights = [np.random(y,x ) for x, y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j},: {self.evaluate(test_data)}/{n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x,y)
            new_b=[newb+deltb for newb, deltb in zip(new_b, delta_b)]
            new_w=[neww+deltw for neww, deltw in zip(new_w, delta_w)]
        self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, new_b)]        
        self.weights=[w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, new_w)]




def sigmoid(z):
    return 1.0/(1.0+ np.exp(-z))


