"""
This code is using a slight modification of the stacked auto-enoder code
to model the relationship between ASINs and simliar products for the Amazon
dataset.

The assumption is that the file is given a sparse matrix representation of 
the data.  The format is as follows;

{"total": 721342}
{"087341764X": {"y": [63021, 324812, 327060, 395580, 395592], "x": 326492}}
{"0873417690": {"y": [141232, 218911, 352484, 352493, 367812], "x": 326493}}
...

The first line represents the number of unique ASINs and is used to size the
feature vectors.  The code will then go through each subsequent line in the
file and construct a sparse matrix representation of the data.

The data is then fed through the auto-encoder with the objective of learning
the weights such that f(x) can closely predict y, where x and y are of 
dimension "total" as stated in the first line of the file.

"""

import os
import sys
import time
import json
import heapq        

import numpy
import scipy
import scipy.sparse

from theano import sparse

import theano
import theano.sparse.basic as basic
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class dA(object):
    """Denoising Auto-Encoder class (dA)

        This class is pretty much ripped off the Theano site.
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bvis=None, output=None):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        Arguments:
                        
            numpy_rng: number random generator used to generate weights of
                type numpy.random.RandomState
                
            theano_rng: Theano random generator; if None is given one is
                generated based on a seed drawn from `rng` of type 
                theano.tensor.shared_randomstreams.RandomStreams
                
            input: a symbolic description of the input or None for
                standalone dA of type theano.tensor.TensorType.
                
            n_visible: number of visible units of type int
    
            n_hidden: number of hidden units of type int
                
            W: Theano variable pointing to a set of weights that should be
                shared belong the dA and another architecture; if dA should
                be standalone set this to None. It is of type 
                theano.tensor.TensorType.
    
            bvis: Theano variable pointing to a set of biases values (for
                visible units) that should be shared belong dA and another
                architecture; if dA should be standalone set this to None.
                It is of type theano.tensor.TensorType


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 name="bvis",
                                 borrow=True)

        self.W = W
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = sparse.csc_matrix(name='input', dtype=numpy.float32) 
            self.y = sparse.csc_matrix(name='output', dtype=numpy.float32)            
        else:
            self.x = input
            self.y = output
            
        #self.params = [self.W, self.b, self.b_prime]
        self.params = [self.W, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1 - corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        
        Arguments:
            input: the piece of data to corrput.
            corruption_level: the degree to which the data should be
                corrupted.
        
        Returns:
            The corrupted data.
            
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer.
        
            Arguments:
                input: the data to feed through the input layer to the
                    hidden layer.
            
            Returns:
                The encoded (lower dimensional) feature vector for a given 
                asin in the form of a 1 hot encoding vector.
        """
        
        return basic.dot(input, self.W)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
            hidden layer.
            
            Arguments:
                hidden: the output from the hidden unit. In the case of this
                    example, it represents the feature vectors for a set of
                    ASINs. these feature vectors are a lower dimensional
                    representation of the data.        
        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
            step of the dA 
            
            Aguments:
                corruption_level: the degree to which teh input should be 
                    corrupted. currently the data is not corrupted.
                
                learning_rate : the degree to which the weights should be
                    altered after a step of gradient descent.
                    
            Returns:
                The errors and the updates to the parameters.
            
        """        
        tilde_x = self.x
        #tilde_x = self.get_corrupted_input(self.x, corruption_level)        
        h = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(h)
        
        # in order to calculate L below, i could not use a sparse matrix
        # for self.y. as a result, i had to convert it to a dense matrix
        # and operate on this new matrix. 
        # TODO: try to make it work with sparse matrix?
        y_mat = sparse.dense_from_sparse(self.y)
        
        # this is the cross-entropy error 
        L = - T.sum(y_mat * T.log(z) + (1.0 - y_mat) * T.log(1.0 - z), axis=1)
        
        # TODO: add L1 or L2 penalization here?
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)

        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
        
def load_feature(path_to_file, callback):
    """This function will load the features, parse them out, and return
        to the caller for processing.
        
        Args: 
            path_to_file : path to the feature vectors file.
            
        Return:
            An array corresponding to the corresponding to a parsed record.        
    """
    
    count = 0
    with open(path_to_file, "r") as fileHandle:
        while True:
            record = fileHandle.readline()
            if not record:
                break
            
            # call the callback function to retrieve the value of interest 
            # to return to the caller.             
            key, value = callback(record, count)
            count = count + 1
            
            yield key, value
    

def generate_sparse_feature(x, y, total):    
    """This function will generate the sparse matrix for a given minibath
        and return to the caller for processing.
        
        Args: 
            x : the input sample. this should only consist of a list of one 
                element since we are creating a one-hot-encoding 
                representation of the data.
            y: the output data that we want to predict. this should be a 
                list of list of elements. that is each ASIN has a list of
                similar items.
            total: the length of each feature vector (the unique # of ASINs)
            
        Return:
            A sparse matrix representation of the mini-batch to be passed to
                the autoencoder.
    """
    
    if len(x) != len(y):
        raise Exception("dimensions are not the same.")

    def generate_attr_vec(idxVector):
        row = []
        col = []
        data = []
 
        for idx, val in enumerate(idxVector):
            if type(val) == list:
                row.extend([idx for c in val])
                col.extend([c for c in val])
                data.extend([1.0 for c in val])
            else:
                row.append(idx)
                col.append(val)
                data.append(1.0)                
                
        return row, col, data

    rowL, colL, dataL = generate_attr_vec(x)         
    rowS, colS, dataS = generate_attr_vec(y)

    sampleMatrix = scipy.sparse.coo_matrix( (dataL, (rowL, colL)), shape=(len(x), total), dtype=numpy.float32)
    labelMatrix = scipy.sparse.coo_matrix( (dataS, (rowS, colS)), shape=(len(x), total), dtype=numpy.float32)

    
    return sampleMatrix, labelMatrix
    
def test_dA(dataset, learning_rate=0.1, training_epochs=15, batch_size=20, 
            n_visible=58, n_hidden=100, modulo=100):

    """
        Function to run the auto-encoder on the data.
        
        Arguments:
            learning_rate: learning rate used for training the DeNosing
                AutoEncoder.
                
            training_epochs: number of epochs used for training
            
            dataset: path to the picked dataset. this path is used to 
                construct the output path for the learned parameters of
                the neural network.
                
            batch_size: the size of the mini batches.
            
            n_visible: the size of the visible units (i.e. the size of the
                input)
            
            n_hidden: the size of the learned representation. in this case
                it corresponds to the size of the distributed representation
                for each ASIN
                
            modulo: helps with debugging.
    """
    
    sample = []
    label = []
    def generate_feature(record, count):
        jsonObject = json.loads(record)
        if "total" in jsonObject:
            return jsonObject["total"], None
        else:
            coords = jsonObject.values()[0]
            return coords["x"], coords["y"]

    # create the generator for leading features                            
    feature_generator = load_feature(dataset, generate_feature)    
    total, _ = feature_generator.next()    
    
    sample = sparse.csc_matrix(name='s', dtype='float32') 
    label = sparse.csc_matrix(name='l', dtype='float32')      
    x = sparse.csc_matrix(name='x', dtype='float32')  
    y = sparse.csc_matrix(name='y', dtype='float32')  

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=n_visible, n_hidden=n_hidden, output=y)
    
    cost, updates = da.get_cost_updates(corruption_level=0.,
                                        learning_rate=learning_rate)

    # this is the function that theano will call to optimize the cost
    # function.
    train_da = theano.function([sample, label], cost, updates=updates,
         givens={x: sample,
                 y: label})

    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    start_time = time.clock()    
    
    epoch = 0
    for epoch in xrange(training_epochs):

        c = []
        miniBatchIn = []
        miniBatchOut = []
        count = 0
        while True:
            try:
                
                # accumulate a mini-batch size worth of data first
                recX, recY = feature_generator.next()                
                miniBatchIn.append(recX)
                miniBatchOut.append(recY)
                count = count + 1
                
                # if we've accumulated enough samples, construct the sparse
                # matrix representation and call theano to train on it
                if count % MINI_BATCH_SIZE == 0:                    
                    samples, labels = generate_sparse_feature(miniBatchIn, miniBatchOut, total)            
                    c.append(train_da(samples, labels))                            
                    
                    miniBatchIn = []
                    miniBatchOut = []   
                    
                if count % modulo == 0:                 
                    print "processed " + str(count) + " training samples"
                    
            except StopIteration:
                # if we reached the end of the file before generating a 
                # a mini-batch, check to see if we've got something in
                # our vector. if we do, train the net on that and then
                # start re-reading the data from the beginning.
                if miniBatchIn:
                    samples, labels = generate_sparse_feature(miniBatchIn, miniBatchOut, total)            
                    c.append(train_da(samples, labels))        

                feature_generator = load_feature(dataset, generate_feature)    
                # skip the total json object                
                feature_generator.next()
                    
                break        

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # we want to write out the parameters now, so that we can read it in
    # later and experiment with it.
    with open(get_params_file_path(dataset), "w") as fileHandle:   
        numpy.savez(fileHandle, W = da.W.get_value(), b = da.b_prime.get_value())

    
def get_params_file_path(feature_file):    
    """
        Function to construct the path to save the learned parameters to.
    """
    return os.path.join(DATA_DIR_PATH, os.path.basename(
        feature_file).split(".")[0] + "_params.out")

def load_params_file(feature_file):
    """
        Function to load the learned parameters to memory.
    """
    
    params_file_path = get_params_file_path(feature_file)
    
    infile = numpy.load(params_file_path)
    return infile

def convert_to_unit_vec(arr):
    """
        Takes a vector and converts to a unit vector.
        
        Arguments:
            arr: the vector to convert.
            
        Returns:
            A unit vector representation of the input.
    """
    
    floatVec = [float(el) for el in arr]
    # calculate the magnitude of the vector
    mag = numpy.sqrt(numpy.dot(floatVec, floatVec))
    
    # divide our array by the magnitude to get our unit vector
    unitVec = floatVec/mag
    return unitVec
    
def most_similar(feature_file, idx):
    """
        Find similar vectors for a given vector by computing the cosine
            similarity and finding the top 5.
        
        Arguments:
            feature_file: the path to the learned weights file.

            idx: the index corresponding to vector that we want to find
                to find similar vectors for.                
                
        Assumptions:
            Assumes that a parameter "W" exists in the file.
    """    
    
    print "loading params file..."
    params = load_params_file(feature_file)
    encodingMatrix = params['W']

    if idx < 0 or idx > encodingMatrix.shape[0]:
        print "not valid dimension..."
        return None
    
    # convert the vector of interest to a unit vector first
    unitVec = convert_to_unit_vec(encodingMatrix[idx])
    
    # now go through every other vector and compute the dot product with it
    # to find the projection length of the vector to our unit vector. then
    # put them into a heapq, so we can quickly retrieve them afterwards.
    heap = []
    for item in range(0, encodingMatrix.shape[0]):
        if item == idx:
            continue
        
        distance = numpy.dot(encodingMatrix[item], unitVec)
        heapq.heappush(heap, (-distance, item))
        
    print "top 5 closest indicies to %s..." % (str(idx))    
    for item in range(0, 6):        
        angle, idx = heapq.heappop(heap)

        if item == idx:
            continue        
        print (-angle, idx)
    
if __name__ == '__main__':    
    if len(sys.argv) > 1:
        path_to_file = sys.argv[1]
    else:
        print "need to provide feature file."
    
    DATA_DIR_PATH = os.path.dirname(path_to_file)    
    NUM_DOCS = 9
    MINI_BATCH_SIZE = 1
                
    if not os.path.exists(get_params_file_path(path_to_file)):
        test_dA(batch_size = MINI_BATCH_SIZE, training_epochs=20, \
            dataset = path_to_file, n_visible=58, n_hidden=100)
        #test_dA(dataset = path_to_file, training_epochs=20, n_visible=58, n_hidden=100)
    else:
        for doc in range(0, NUM_DOCS):
            most_similar(path_to_file, doc)