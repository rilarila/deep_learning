from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy
import pandas

import theano
import theano.tensor as T


import deep_learning

class MLP(deep_learning.Model):
    def __init__(self,n_in,n_out,features,rng,L1_reg=0.00,L2_reg=0.0001):
        if n_out > 1:
            self.__class__ = deep_learning.Classifier
        else:
            n_out = 1
            self.__class__ = deep_learning.Regression

        L1 = 0
        L2 = 0

        self.params = []

        input = self.get_input()
	layer_n_in = n_in

        for layer_n_out in features:
            layer = deep_learning.Layer(input,layer_n_in,layer_n_out,activation=T.tanh,rng=rng)
            self.params += layer.params

            L1 += layer.L1
            L2 += layer.L2

            input = layer.output
            layer_n_in = layer_n_out

        self.output_layer = deep_learning.Layer(input, layer_n_in, n_out)

        self.params += self.output_layer.params

        L1 += self.output_layer.L1
        L2 += self.output_layer.L2

        cost_addition = L1_reg*L1 + L2_reg*L2

        self.build(cost_addition)

def test_mlp(n_out = 10, features=[500], learning_rate=0.01, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20):
    datasets = deep_learning.load_data(dataset)

    n_in = datasets[0][0].shape[1].eval()

    rng = numpy.random.RandomState(1234)
    model = MLP(n_in, n_out, features, rng=rng)

    deep_learning.train(model,learning_rate,datasets,batch_size,n_epochs)

def test_prediction(classifier_dump='best_model.pkl', dataset='mnist.pkl.gz',num=10):
    results = deep_learning.predict(classifier_dump,dataset,num)
    df = pandas.DataFrame(results).T
    print(df.to_string(header=False))
    return results

if __name__ == '__main__':
#    test_mlp(n_out = 10,n_epochs=10,features=[500,20])
    results = test_prediction(num=30)
