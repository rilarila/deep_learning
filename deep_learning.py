from __future__ import print_function

import six.moves.cPickle as pickle
import csv
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

class MLP(object):
	def __init__(self,n_in,n_out,features=[],rng=None,L_reg=[0.00,0.0001],learning_rate=0.01):
		print('... building the model')

		self.L_reg = numpy.array(L_reg)
		self.params = []
		self.input_variable = T.matrix('x')
		self.output_variable = T.vector('y')
		self.cost = 0

		self.layer_n_out = n_in
		self.layer_output = self.input_variable

		self.learning_rate = learning_rate

		if rng is None:
			rng = numpy.random.RandomState(1234)
		self.rng = rng

		for layer_n_out in features:
			self.add_layer(layer_n_out,T.tanh)

		self.add_layer(n_out)

	def add_layer(self,n_out,activation=None):
		if activation is not None:
			layer = Layer(self.layer_output,self.layer_n_out,n_out,activation=activation,rng=self.rng)
		else:
			layer = Layer(self.layer_output,self.layer_n_out,n_out)

		self.params += layer.params
		self.cost += sum(self.L_reg * layer.L)
		self.layer_output = layer.output
		self.layer_n_out = n_out

		self.output_layer = layer

	def get_updates(self):
		return [(param, param - self.learning_rate * T.grad(self.cost,param)) for param in self.params]

	def train(self,data,batch_size,max_epochs,distribution={'train':5./7,'valid':1./7,'test':1./7},model_dump='best_model.pkl'):
		print('... training the model')

		def partition(data,distribution):
			s = sum(distribution.values())
			distribution = {i:distribution[i]/float(s) for i in distribution}

			def shared_dataset(data, borrow=True):
				shared_x = theano.shared(data[:,:-1],borrow=borrow)
				shared_y = theano.shared(data[:,-1],borrow=borrow)
				return {'x':shared_x, 'y':shared_y,'num':len(data)}

			partitioned_data = {}
			n = len(data)
			start = 0
			for i in distribution:
				end = int(start + distribution[i]*n)
				partitioned_data[i] = shared_dataset(data[start:end])
				start = end
			return partitioned_data

		partitioned_data = partition(data,distribution) 

		n_train_batches = partitioned_data['train']['num'] // batch_size
		n_valid_batches = partitioned_data['valid']['num'] // batch_size
		n_test_batches  = partitioned_data['test'] ['num'] // batch_size

		index = T.lscalar() 

		def getGivens(dataset):
			return {
				self.input_variable:  dataset['x'][index * batch_size: (index + 1) * batch_size],
				self.output_variable: dataset['y'][index * batch_size: (index + 1) * batch_size]
			}

		test_model = theano.function(
			inputs=[index],
			outputs=self.score,
			givens=getGivens(partitioned_data['test'])
		)

		validate_model = theano.function(
			inputs=[index],
			outputs=self.score,
			givens=getGivens(partitioned_data['valid'])
		)

		train_model = theano.function(
			inputs=[index],
			updates=self.get_updates(),
			givens=getGivens(partitioned_data['train'])
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

				if n_test_batches > 0:
					test_scores = [test_model(i) for i in range(n_test_batches)]
					test_score = numpy.mean(test_scores)

					print('\tepoch %i, test score of best model %f%%' % (epoch+1, test_score*100.))

				self.dump(model_dump)

			if epoch > n_epochs:
				break

		end_time = timeit.default_timer()

		print('Optimization complete with best validation score of %f%%, test score %f%%' % (best_validation_score * 100., test_score * 100))

		print('The code run for %d epochs, with %f epochs/sec' % (epoch+1, (epoch+1.) / (end_time - start_time)))
		print('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time))

	def predict(self,data,d={}): 
		predict_model = theano.function( 
			inputs=[self.input_variable], 
			outputs=self.output)

		def f(bin_num):
			if bin_num in d:
				return d[bin_num]
			return bin_num

		f = numpy.vectorize(f)

		x = data[:,:-1]
		y = f(data[:,-1])

		predicted_values = f(predict_model(x))

		results = {'Predicted values':predicted_values,'Actual values': y}

		return results

	def dump(self,model_dump):
		with open(model_dump,'wb') as f:
			pickle.dump(self,f)

class Regression(MLP):
	def __init__(self,n_in,features=[],rng=None,L_reg=[0.00,0.0001],learning_rate=0.01):
		super(self.__class__, self).__init__(n_in,1,features,rng,L_reg,learning_rate)

		self.output = self.output_layer.output.flatten()

		base_cost = T.mean((self.output - self.output_variable)**2)
		param = 1

		self.cost += base_cost
		self.score = (param/(base_cost+param))

class Classification(MLP):
	def __init__(self,n_in,n_out,features=[],rng=None,L_reg=[0.00,0.0001],learning_rate=0.01):
		super(self.__class__, self).__init__(n_in,n_out,features,rng,L_reg,learning_rate)

		p_y_given_x = T.nnet.softmax(self.output_layer.output)

		self.output = T.argmax(p_y_given_x, axis=1)
		self.cost += -T.mean(T.log(p_y_given_x)[T.arange(self.output_variable.shape[0]), self.output_variable.astype('int32')])
		self.score = T.mean(T.eq(self.output, self.output_variable))

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

def format_data(raw_data,regression):

	def unlabel(row):
		try:
			return row.astype('float')
		except ValueError:
			labels = list(set(row))
			shift = int(2**(len(labels).bit_length()))
			d = {labels[i]:[int(j) for j in bin(shift+i)[3:]] for i in range(len(labels))}
			'''
			d = {labels[i]:[int(j==i) for j in range(len(labels))] for i in range(len(labels))}
			'''
			return numpy.array([d[i] for i in row])
			

	x = numpy.column_stack([unlabel(i) for i in raw_data[:,:-1].T])

	if regression:
		d = {}
		y = raw_data[:,-1].astype('float')
		n_out = 1

	else:
		raw_y = raw_data[:,-1]
		values = list(set(raw_y))
		d_reverse = {values[i]:i for i in range(len(values))}
		y = [d_reverse[i] for i in raw_y]
		d = {j:i for (i,j) in d_reverse.items()}
		n_out = len(d)

	data = numpy.column_stack([x, y])

	return (data, d, n_out)

def load_data(dataset,regression,header=True,output_value_is_first=False,upsampling=False):
	print('... loading data')

	if os.path.split(dataset)[-1] == 'mnist.pkl.gz':
		with gzip.open(dataset, 'rb') as f:
			loaded = pickle.load(f)

		data = numpy.vstack([numpy.c_[i] for i in loaded])
		d = {}
		n_out = 1 if regression else 10

	else:
		with open(dataset) as f:
			values = [i for i in csv.reader(f)]

		if header:
			values = values[1:]

		raw_data = numpy.array(values)

		if output_value_is_first:
			raw_data = numpy.roll(raw_data,-1,1)

		data, d, n_out = format_data(raw_data,regression)

		if upsampling and not regression:
			freq = Counter(data[:,-1])
			n = max(freq.values())
			

	return {'data':data,'d':d,'n_out':n_out}

def load_model(model_dump):
	with open(model_dump,'rb') as f:
		model = pickle.load(f)

	return model
