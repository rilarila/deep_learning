ej sucks

## deep\_learning documentation

### Typical use:
```python
import pandas


dataset = 'mnist.pkl.gz'
loaded_data = load_data(dataset,regression=False)
data = loaded_data['data']

n_in = data.shape[1]-1
n_out = loaded_data['n_out']

model = Classification(n_in,n_out,[50,20])
model.train(data,batch_size = 20, max_epochs = 100)

results = model.predict(data[:100])
print(pandas.DataFrame(results).T)




dataset = 'heart.csv'
loaded_data = load_data(dataset,regression=True,header=True)
data = loaded_data['data']

n_in = data.shape[1]-1

model = Regression(n_in,[10])
model.train(data,batch_size = 10, max_epochs = 1000)

results = model.predict(data[:100],loaded_data['d'])
print(pandas.DataFrame(results).T)




dataset = 'credit.csv'
output = easy_deep_learning(dataset,regression=False,depth=3,max_epochs=5,predict_count=100)

```

### Classes and Functions:
```python
class MLP(n_in,n_out,features=[],rng=None,L_reg=numpy.array([0.00,0.0001]),learning_rate=0.01)
	abstract class

	args
		n_in - int, input values per data point
		n_out - int, if n_out==1, performs regression, otherwise performs classification into n_out bins
		features - array<int>, size of each hidden layer
		rng - numpy random state
		L_reg - array used to calculate cost
			cost += L_reg * layer.L
		learning_rate - float, multiplicative factor of gradient when updating parameters

	attributes
		L_reg - numpy array<float>, coefficients for L of each layer when calculating cost
		params - array<theano symbolic tensor>, flattened list of parameters for each layer
		input_variable - theano symbolic matrix
		output_variable - theano symbolic vecotr
		cost - theano expression, cost to perform gradient training
		layer_n_out - int, values per data point for final layer
		layer_output - theano symbolic tensor, output of final layer
		learning_rate - float, multiplicative factor of gradient when updating parameters
		rng - numpy random state

	methods
		get_updates() - array<theano symbolic tensor, theano expression>, updates calculated using gradient of each parameter with respect to cost

		add_layer(n_out,activation=None)
			n_out - int, size of output of each data point for this layer
			activation - function(theano tensor), layer activation
                        
			updates model params, cost, layer_output, layer_n_out, output_layer

		train(data,batch_size,max_epochs,distribution={'train':5./7,'valid':1./7,'test':1./7},model_dump='best_model.pkl') - trains the model using the provided data

			args
				data - numpy matrix<float>
				batch_size - int
				max_epochs - int
				distribution - dictionary<(train|valid|test),float proportion>, partitions data
				model_dump - string, path to save representation of model

		predict(data,d={})
			returns dictionary of the model's predicted values using input x, along with the actual values y

			args
        		        data - numpy matrix<float>, each row is the data point followed by the true value
				d - dictionary<int,string> transformation of bin to label

		#### TODO - currently just uses python pickle
		dump(model_dump):`
			saves the model to a text file at location model_dump

			args
				model_dump - string, path to save model


class Regression(n_in,features=[],rng=None,L_reg=[0.00,0.001],learning_rate=0.01):
	subclass of MLP

	attributes
		output - theano symbolic vector of output values
		cost - theano symbolic float
		score - theano symbolic float


class Classification(n_in,n_out,features=[],rng=None,L_reg=[0.00,0.001],learning_rate=0.01):
	subclass of MLP

	attributes		
		output - theano symbolic vector of output values
		cost - theano symbolic float
		score - theano symbolic float


class Layer(input, n_in, n_out, activation=(lambda x: x), rng=None):
	args
		input - theano symbolic matrix
		n_in - int
		n_out - int
		activation - function(theano tensor)
		rng - numpy random state

	attributes
		params - array<theano symbolic tensor>
		output - theano symbolic tensor
		L - numpy array<theano expression>
			[abs(W).sum(), (W ** 2).sum()]


def format_data(raw_data,regression):
        formats data to contain only floats to prepare for training
	returns tuple (data,d,n_out)
		data - numpy matrix of floats
		d - dictionary to transform bin number to label in the case of classification (empty in the case of regression)
		n_out - number of bins in the case of classification (1 in the case of regression)

	args
		raw_data - numpy matrix where each row contains the data point followed by the true value
		regression - boolean representing model type


def load_data(dataset,regression,header=True,true_value_is_last=True,upsampling=False)
	returns dictionary{'data': numpy matrix<float> data, d: dictionary<int,string> transformation of bin to label,'n_in': int number input features, 'n_out': number of output bins, or 1 for regression}

	args
		datset - string, path to dataset csv
		regression - boolean, classification or regression
		header - boolean, whether or not to ignore first line
		output_value_is_first - boolean, true if output is the first value, false if the output is the last value in each row
		upsampling - boolean, if true and the model is classification, upsamples data to have an equal number of data points for each bin


#### TODO - currently only uses python pickle
def load_model(model_dump):
	returns model saved using dump_model at location model_dump

	args
		model_dump - string, path to load model
```
=======
# deep_learning
