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
    def updates(self,learning_rate):
        return [(param, param - learning_rate * T.grad(self.cost,param)) for param in self.params]

    def get_input(self):
        print('... building the model')

        self.input_variable = T.matrix('x')
        self.output_variable = T.ivector('y')

        return self.input_variable

    def build(self,cost_addition=0):
        self.output = self.get_output()
        self.base_cost = self.get_base_cost()
        self.cost = self.base_cost+cost_addition
        self.score = self.get_score()

class Regression(Model):
    def get_output(self):
        return self.output_layer.output.flatten()

    def get_base_cost(self):
        return T.mean(abs(self.output - self.output_variable))

    def get_score(self):
        param = 1.0
        return T.mean(param/(self.base_cost+param))

class Classifier(Model):
    def get_p_y_given_x(self):
        if not hasattr(self,'p_y_given_x'):
            self.p_y_given_x = T.nnet.softmax(self.output_layer.output)
        return self.p_y_given_x

    def get_output(self):
        return T.argmax(self.get_p_y_given_x(), axis=1)

    def get_base_cost(self):
        return -T.mean(T.log(self.get_p_y_given_x())[T.arange(self.output_variable.shape[0]), self.output_variable])

    def get_score(self):
        return T.mean(T.eq(self.output, self.output_variable))

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

        self.L1 = abs(W).sum()
        self.L2 = (W ** 2).sum()


def load_data(dataset,num=None):
    data_dir, data_file = os.path.split(dataset)
    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        sets = pickle.load(f)
        if num != None:
            sets = [[j[:num] for j in i] for i in sets]

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    return [shared_dataset(i) for i in sets]

def train(classifier,learning_rate,datasets,batch_size,n_epochs,classifier_dump='best_model.pkl'):
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant


    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set[1].shape[0].eval() // batch_size
    n_valid_batches = valid_set[1].shape[0].eval() // batch_size
    n_test_batches = test_set[1].shape[0].eval() // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    def getGivens(dataset):
        return {
            classifier.input_variable: dataset[0][index * batch_size: (index + 1) * batch_size],
            classifier.output_variable: dataset[1][index * batch_size: (index + 1) * batch_size]
        }

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.score,
        givens=getGivens(test_set)
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.score,
        givens=getGivens(valid_set)
    )

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=classifier.cost,
        updates=classifier.updates(learning_rate),
        givens=getGivens(train_set)
    )
    # end-snippet-3

    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_score = 0.
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_score = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_score = numpy.mean(validation_score)

                print(
                    'epoch %i, minibatch %i/%i, validation score %f%%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_score * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_score > best_validation_score:
                    #improve patience if loss improvement is good enough
                    if this_validation_score > best_validation_score *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_score = this_validation_score
                    # test it on the test set

                    test_scores = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_scores)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test score of'
                            ' best model %f%%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open(classifier_dump, 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f%%, '
            'with test performance %f%%'
        )
        % (best_validation_score * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict(classifier_dump='best_model.pkl', dataset='mnist.pkl.gz',num=10): 
    classifier = pickle.load(open(classifier_dump))
 
    predict_model = theano.function( 
        inputs=[classifier.input_variable], 
        outputs=classifier.output) 
 
    test_set_x, test_set_y = load_data(dataset,num)[2] 

    predicted_values = predict_model(test_set_x.eval()) 
    actual_values = test_set_y.eval()

    return {'Predicted values':predicted_values,'Actual values':actual_values}
