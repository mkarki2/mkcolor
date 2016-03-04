"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.gof import graph


class LinearRegression(object):
    """
    The Linear Regression layer. It's similar to LogisticRegression,
    but we will only have one output layer, and we don't use their 'errors' method.
    """

    def __init__(self, input, n_in, n_out):
        """
        :input: A symbolic variable that describes the input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value = numpy.zeros( (n_in, n_out), dtype=theano.config.floatX ),
                               name = 'W',
                               borrow = True)
        self.b = theano.shared(value = numpy.zeros( (n_out,), dtype=theano.config.floatX ),
                               name = 'b',
                               borrow = True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.p_y_given_x[:,0]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input



    def squared_errors(self, y):
        """ Returns the mean of squared errors of the linear regression on this data. """
        return T.mean((self.y_pred - y) ** 2)


def load_data(dataset):
    data = []
    import re

    if isinstance(dataset, str) and re.search('\.csv$', dataset):
        with open(dataset) as f:
            f.readline()
            for line in f:
                if line=='\n':
                    continue
                data.append([float(x.strip()) for x in line.strip().split(',')])

        train_x = [data[x][:-1] for x in range(int(len(dataset)*.7))]
        train_y = [data[x][-1] for x in range(int(len(dataset)*.7))]

        valid_x = [data[x][:-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]
        valid_y = [data[x][-1] for x in range(int(len(dataset)*.7), int(len(dataset)*.85))]

        test_x = [data[x][:-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]
        test_y = [data[x][-1] for x in range(int(len(dataset)*.85), int(len(dataset)))]

        train_set = (numpy.array(train_x), numpy.array(train_y))
        valid_set = (numpy.array(valid_x), numpy.array(valid_y))
        test_set = (numpy.array(test_x), numpy.array(test_y))

    else:
        """
        Copying this from documentation online, including some of the nested 'shared_dataset' function,
        but I'm also returning the number of features, since it's easiest to detect that here.
        """
        train_set, valid_set,test_set = dataset[0], dataset[1], dataset[2]
        assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], \
            "Number of features for train,val do not match: {} and {}.".format(train_set.shape[1],valid_set.shape[1])


    def shared_dataset(data_xy, borrow=True):
            """
            Function that loads the dataset into shared variables. It is DIFFERENT from the online
            documentation since we can keep shared_y as floats; we won't be needing them as indices.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    num_features = (train_set[0].shape)[1]
    rval = [(train_set_x,train_set_y), (valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return rval,num_features


def optimization_mnist(learning_rate=0.01,n_epochs=1000,
                           dataset='',
                           batch_size=10):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets,features = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    numpy_rng = numpy.random.RandomState(123)
    print ('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.vector('y')  # labels, presented as 1D vector of labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LinearRegression(input=x, n_in=features, n_out=1)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.squared_errors(y) #+ L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # Compiling a Theano function that computes the error by the model on a minibatch. Since we
    # don't have simple classification, just return the classifier.squared_errors().
    validate_model = theano.function([index],
                                     cost,
                                     givens = {
                                        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                                     })

    # Compute the gradient of the cost w.r.t. parameters; code matches the documentation.
    gparams = [T.grad(cost, param) for param in classifier.params]

    # How to update the model parameters as list of (variable, update_expression) pairs.
    updates = [ (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]

    # Compiling a Theano function `train_model` that returns the cost AND updates parameters.
    train_model = theano.function(inputs = [index],
                                  outputs = cost,
                                  updates = updates,
                                  givens = {
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]
                                  })

    test_model = theano.function(inputs = [index],
                                 outputs = cost,
                                 givens = {
                                    x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                    y: test_set_y[index * batch_size:(index + 1) * batch_size]
                                 })
    ###############
    # TRAIN MODEL #
    ###############
    print ('... training the model')
    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    # Other variables of interest
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(int(n_train_batches)):

            # Training.
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            # Evaluate on validation set
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(int(n_valid_batches))]
                this_validation_loss = numpy.mean(validation_losses)
                print ("epoch {}, minibatch {}/{}, validation MAE {:.5f}".format(epoch,
                                minibatch_index + 1, n_train_batches, this_validation_loss))

                # If best valid so far, improve patience and update the 'best' variables.
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # save the best model
                    with open('best_model_linear.pkl', 'w') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print ("Optimization complete.")



def predict():
    """
    An example of how to load a train model and use it
    to predict labels.
    """

    # load the saved model
    classifier = cPickle.load(open('best_model_linear.pkl'))
    y_pred = classifier.y_pred

    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']

    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)

    X_test = numpy.random.rand(1000,500)*.75 +.25
    X_test=  numpy.append(X_test, numpy.random.rand(1000,500)*.75,axis=0)

    y_test = numpy.random.rand(1000,)*.3
    y_test = numpy.append(y_test, numpy.random.rand(1000,)*.3+.7,axis=0)

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print (predicted_values)


if __name__ == '__main__':
    X_train = numpy.random.rand(5000,500)*.75 +.25
    X_train = numpy.append(X_train,numpy.random.rand(5000,500)*.75,axis=0)

    y_train = numpy.random.rand(5000,)*.3
    y_train = numpy.append(y_train,numpy.random.rand(5000,)*.3+.7,axis=0)

    X_val = numpy.random.rand(1000,500)*.75 +.25
    X_val = numpy.append(X_val,numpy.random.rand(1000,500)*.75,axis=0)

    y_val = numpy.random.rand(1000,)*.3
    y_val = numpy.append(y_val,numpy.random.rand(1000,)*.3+.7,axis=0)

    X_test = numpy.random.rand(1000,500)*.75 +.25
    X_test = numpy.append(X_test,numpy.random.rand(1000,500)*.75,axis=0)

    y_test = numpy.random.rand(1000,)*.3
    y_test = numpy.append(y_test,numpy.random.rand(1000,)*.3+.7,axis=0)
    data = [ (X_train,y_train) , (X_val,y_val),(X_test,y_test) ]
    optimization_mnist(dataset=data)
