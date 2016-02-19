from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class Model(object):
    def __init__(self,n_in,learning_rate):
        print('... building the model')

        self.L = numpy.zeros(2,dtype=object)
        self.params = []
        self.input_variable = T.matrix('x')
        self.output_variable = T.ivector('y')
        self.cost = 0

        self.n_in = n_in
        self.input = self.input_variable

        self.learning_rate = learning_rate

    def get_updates(self):
        return [(param, param - self.learning_rate * T.grad(self.cost,param)) for param in self.params]

def Regression(network):
    network.output = network.output_layer.output.flatten()

    base_cost = T.mean(abs(network.output - network.output_variable))
    param = 1

    network.cost += base_cost
    network.score = T.mean(param/(base_cost+param))

def Classifier(network):
    p_y_given_x = T.nnet.softmax(network.output_layer.output)

    network.output = T.argmax(p_y_given_x, axis=1)
    network.cost += -T.mean(T.log(p_y_given_x)[T.arange(network.output_variable.shape[0]), network.output_variable])
    network.score = T.mean(T.eq(network.output, network.output_variable))

class Layer(object):
    def __init__(self, input, n_in, n_out, activation=(lambda x: x), rng=None):
        if rng is None:
            W_values = numpy.zeros((n_in,n_out),dtype=theano.config.floatX)
        else:
            bound = numpy.sqrt(6. / (n_in + n_out))
            W_values = numpy.asarray(rng.uniform(low=-bound,high=bound,size=(n_in,n_out)))

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

        b_values = numpy.zeros((n_out,),dtype=theano.config.floatX)

        W = theano.shared(value=W_values, name='W', borrow=True)

        b = theano.shared(value=b_values, name='b', borrow=True)

        self.params = [W,b]

        self.output = activation(T.dot(input, W) + b)

        self.L = numpy.array([abs(W).sum(),(W ** 2).sum()])


def load_data(dataset,num=None,distribution=[5,1,1]):
    print('... loading data')

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        return {'x':shared_x, 'y':T.cast(shared_y, 'int32'),'num':len(data_x)}

    with gzip.open(dataset, 'rb') as f:
        sets = pickle.load(f)
        data = {}
        if os.path.split(dataset)[-1] == 'mnist.pkl.gz':
            if num != None:
                sets = [[j[:num] for j in i] for i in sets]

            data['train'] = shared_dataset(sets[0])
            data['valid'] = shared_dataset(sets[1])
            data['test']  = shared_dataset(sets[2])

            data['n_in'] = len(sets[0][0][0])

        else:
            raise NotImplementedError('other data source: %s' % dataset)

    return data

def train(model,data,batch_size,max_epochs,model_dump='best_model.pkl'):
    print('... training the model')

    n_train_batches = data['train']['num'] // batch_size
    n_valid_batches = data['valid']['num'] // batch_size
    n_test_batches  = data['test'] ['num'] // batch_size

    index = T.lscalar() 

    def getGivens(dataset):
        return {
            model.input_variable:  dataset['x'][index * batch_size: (index + 1) * batch_size],
            model.output_variable: dataset['y'][index * batch_size: (index + 1) * batch_size]
        }

    test_model = theano.function(
        inputs=[index],
        outputs=model.score,
        givens=getGivens(data['test'])
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=model.score,
        givens=getGivens(data['valid'])
    )

    train_model = theano.function(
        inputs=[index],
        updates=model.get_updates(),
        givens=getGivens(data['train'])
    )

    n_epochs = max_epochs
    epoch_increase = 2
    improvement_threshold = 0.995

    best_validation_score = 0.
    test_score = 0.
    start_time = timeit.default_timer()

    for epoch in range(max_epochs):
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)

        validation_score = numpy.mean([validate_model(i) for i in range(n_valid_batches)])

        print('epoch %i, validation score %f%%' % (epoch+1, validation_score*100.))

        if validation_score > best_validation_score:
            if validation_score > best_validation_score * improvement_threshold:
                n_epochs = max(n_epochs, epoch * epoch_increase)

            best_validation_score = validation_score

            test_scores = [test_model(i)
                                   for i in range(n_test_batches)]
            test_score = numpy.mean(test_scores)

            print('\tepoch %i, test score of best model %f%%' % (epoch+1, test_score*100.))

            with open(model_dump, 'wb') as f:
                    pickle.dump(model, f)

        if epoch > n_epochs:
            break

    end_time = timeit.default_timer()

    print('Optimization complete with best validation score of %f%%, test score %f%%' % (best_validation_score * 100., test_score * 100))

    print('The code run for %d epochs, with %f epochs/sec' % (epoch+1, (epoch+1.) / (end_time - start_time)))
    print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time))

def predict(dataset, model_dump='best_model.pkl',num=10): 
    model = pickle.load(open(model_dump))
 
    predict_model = theano.function( 
        inputs=[model.input_variable], 
        outputs=model.output) 

    test_set = load_data(dataset,num)['test']

    predicted_values = predict_model(test_set['x'].eval()) 
    actual_values = test_set['y'].eval()

    return {'Predicted values':predicted_values,'Actual values':actual_values}
