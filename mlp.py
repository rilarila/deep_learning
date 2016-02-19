from __future__ import print_function

import sys

import numpy
import pandas

import theano.tensor as T


import deep_learning

class MLP(deep_learning.Model):
    def __init__(self,n_in,n_out=1,features=[],rng=None,L1_reg=0.00,L2_reg=0.0001,learning_rate=0.01):

        super(self.__class__, self).__init__(n_in,learning_rate)
        
        if rng is None:
            rng = numpy.random.RandomState(1234)
        self.rng = rng

        for layer_n_out in features:
            self.add_layer(layer_n_out,T.tanh)

        self.add_layer(n_out)
      
        self.cost += sum(self.L * [L1_reg,L2_reg])
        
        if n_out > 1:
            deep_learning.Classifier(self)
        else:
            deep_learning.Regression(self)

    def add_layer(self,n_out,activation=None):
        if activation is not None:
            layer = deep_learning.Layer(self.input,self.n_in,n_out,activation=activation,rng=self.rng)
        else:
            layer = deep_learning.Layer(self.input,self.n_in,n_out)
        self.params += layer.params
        self.L += layer.L
        self.input = layer.output
        self.n_in = n_out

        self.output_layer = layer

def test_mlp(n_out = 10, features=[500], max_epochs=1000, dataset='mnist.pkl.gz', batch_size=20):
    data = deep_learning.load_data(dataset)

    n_in = data['n_in']
    
    model = MLP(n_in, n_out, features)

    deep_learning.train(model,data,batch_size,max_epochs)

def test_prediction(dataset='mnist.pkl.gz',num=10):
    results = deep_learning.predict(dataset,num=num)
    df = pandas.DataFrame(results).T
    print(df.to_string(header=False))
    return results

if __name__ == '__main__':
    test_mlp(n_out = 10,max_epochs=3,features=[50,20])
    test_prediction(num=20)

    test_mlp(n_out = 1,max_epochs=3,features=[50,20])
    test_prediction(num=20)
