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
from optparse import OptionParser

import theano
import theano.sparse.basic as basic
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from auto_encoder_modeler import dA

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.        
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                    high=4 * numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            #if activation == theano.tensor.nnet.sigmoid:
            #    W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is not None:
            if type(b) == numpy.ndarray:
                b_initial = theano.shared(value=b, name="b", borrow=True)            
            else:
                b_initial = b
        else:
            b_initial = None

        #if b is None:
        #    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        #    b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b_initial

        self.activation = activation
        
        self.output = self.get_hidden_values(input)
        
        # parameters of the model
        self.params = [self.W, self.b] if self.b else [self.W]

    def get_hidden_values(self, input):
        
        #if type(input) != basic.SparseVariable:
        if type(input) == T.TensorVariable:
            sparse_input = basic.csc_from_dense(input)
        else:
            sparse_input = input
        #x_is_sparse_variable = basic._is_sparse_variable(input)
        #w_is_sparse_variable = basic._is_sparse_variable(self.W)
    
        #if not x_is_sparse_variable and not w_is_sparse_variable:        
        #    fn = T.dot
        #else:        
        #    fn = basic.dot

        lin_output = basic.dot(sparse_input, self.W)
        if self.activation is None and self.b is None:
            return lin_output
        elif self.activation is None:
            return lin_output + self.b
        elif self.b is None:
            return self.activation(lin_output)
        else:
            return self.activation(lin_output + self.b)
        
class SdA(object):
    """Denoising Auto-Encoder class (dA)

        This class is pretty much ripped off the Theano site.
    """

    def __init__(self, input, label, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], activation=T.nnet.sigmoid, 
                 b=None):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data        
        self.x = input  # the data is presented as rasterized images
        self.y = label  # the labels are presented as 1D vector of
                                 # [int] labels

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
                layer_output = self.y
            else:
                layer_input = self.sigmoid_layers[-1].output
                layer_output = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=activation,
                                        b=b) #T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer                             
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b,
                          output=layer_output,
                          activation=activation)
            self.dA_layers.append(dA_layer)

    def inspect_inputs(self, i, node, fn):
        print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
    
    def inspect_outputs(self, i, node, fn):
        print "output(s) value(s):", [output[0] for output in fn.outputs]

    def pretraining_functions(self, sample, label, learning_rate, corruption_level):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        pretrain_fns = []
        #count = 1
        for i, dA in enumerate(self.dA_layers):
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level=corruption_level,
                                                learning_rate=learning_rate)
            # this is the function that theano will call to optimize the cost
            # function.
            train_da = theano.function([sample, label,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.1)],
                    outputs=cost, 
                    updates=updates,
                    givens={self.x: sample,
                            self.y: label})
                #mode=theano.compile.debugmode.DebugMode())
                #mode=theano.compile.MonitorMode(
                #        pre_func=self.inspect_inputs,
                #        post_func=self.inspect_outputs).excluding('local_elemwise_fusion', 'inplace'))
            
            # append `fn` to the list of functions
            pretrain_fns.append(train_da)

            #fileName = str(count) + "_out"
            #theano.printing.pydotprint(train_da, outfile=fileName)             
            #count = count + 1
            
        return pretrain_fns

    def get_hidden_values(self, data):
        for da in self.dA_layers:
            h = da.get_hidden_values(data)
            data = h

        return data
        
    def get_reconstruction_errors(self, data):
        inputData = data
        # feed it forward        
        data = self.get_hidden_values(data)
                    
        # now reverse
        for da in reversed(self.dA_layers):
            z = da.get_reconstructed_input(data)
            data = z
            
        error = z.eval() - inputData
        return error
        
        
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
    
def test_SdA(dataset, learning_rates=[0.10], training_epochs=15, batch_size=20, 
            n_visible=58, n_hidden=[100], modulo=100, W=None, bhid=None, 
            bvis=None, corruption_levels=[0.0], activation=T.nnet.sigmoid):
                
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
    
    # numpy random generator
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    print '... building the model'
    
    # index to a [mini]batch
    sample = sparse.csc_matrix(name='s', dtype='float32') 
    label = sparse.csc_matrix(name='l', dtype='float32')      
    x = sparse.csc_matrix(name='x', dtype='float32')  
    y = sparse.csc_matrix(name='y', dtype='float32') 

    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use
    
    # construct the stacked denoising autoencoder class
    sda = SdA(numpy_rng=rng, theano_rng=theano_rng, n_ins=n_visible,
              hidden_layers_sizes=n_hidden, input=x, label=y, 
              activation=activation, b=bhid)


    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(sample, label, learning_rate, 
                                                corruption_level)
                                                
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
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    
    start_time = time.clock()    
    
    epoch = 0
    for i in xrange(sda.n_layers):
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
                        error = pretraining_fns[i](samples, labels, 
                            corruption_levels[i], learning_rates[i])
                        c.append(error)                            
                        
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
                        error = pretraining_fns[i](samples, labels, 
                            corruption_levels[i], learning_rates[i])
                        c.append(error)        
    
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
    #with open(get_params_file_path(dataset), "w") as fileHandle:   
    #    numpy.savez(fileHandle, W = da.W.get_value(), b = da.b_prime.get_value())

    
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
    
def most_similar(params, idx, asinTitleMap, asinIdxMap):
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
        
    #print "top 20 closest indicies to %s..." % (str(idx))    
    print "====================== index: " + str(idx)
    for item in range(0, 10):        
        angle, idx = heapq.heappop(heap)

        if item == idx:
            continue      
        
        if asinIdxMap and asinTitleMap:
            if idx in asinIdxMap:
                if asinIdxMap[idx] in asinTitleMap:
                    print (-angle, idx, asinTitleMap[asinIdxMap[idx]])
        else:
            print (-angle, idx)
            

def obtain_key_title_mapping(asinTitlePath):
    
    products = dict()
    for line in open(asinTitlePath, "r"):       
        line = line.replace('"',"")
        key, value = line.rstrip().split("\t")
        products[key] = value

    return products

def load_asin_idx_mapping(pathToFeatures):
    
    def generate_feature(record, count):
        jsonObject = json.loads(record)
        if "total" not in jsonObject:
            key = jsonObject.keys()[0]
            coords = jsonObject.values()[0]
            return key, coords["x"]
        else:
            return None, None

    # create the generator for leading features                            
    feature_generator = load_feature(pathToFeatures, generate_feature)    
    feature_generator.next()    
    
    asinIdxMap = dict()
    while True:
        try:
            key, idx = feature_generator.next()
            asinIdxMap[idx] = key
        except StopIteration:
            break
        
    return asinIdxMap

    
if __name__ == '__main__':    
    parser = OptionParser()
    parser.add_option("-u", "--update", action = "store_true", dest="update",
                      help="update a given model")
    parser.add_option("-f", "--file", dest="file", action="store",
                      help="path to training data", metavar="FILE")
    parser.add_option("-t", "--title-file", dest="titleFile", action="store",
                      help="path to asin to title mapping", metavar="FILE")
    parser.add_option("-w", "--write-reconstruction", dest="reconstruct", 
                      action="store_true", help="write out re-construction of input")
    parser.add_option("-d", "--dense-type", dest="dense", 
                      action="store_true", help="construct features from dense data")
    parser.add_option("-s", "--save-params", dest="params", 
                      action="store_true", help="save parameters after training")
    
    (options, args) = parser.parse_args()

    if options.file:    
        path_to_file = options.file
    else:
        print "need to provide feature file."

    if options.titleFile:
        asin_to_title = options.titleFile
    else:
        asin_to_title = None

    '''
    TODO: finish implementing
    if options.reconstruct:
        reconstruct = True
    else:
        reconstruct = False
        
    if options.dense:
        fn = generate_dense_attr_vec
    else:
        fn = generate_one_hot_encoding_attr_vec
        
    if options.params:
        save_params = True
    else:
        save_params = False
    '''
    
    DATA_DIR_PATH = os.path.dirname(path_to_file)    
    NUM_DOCS = 1
    MINI_BATCH_SIZE = 1

    if options.update:
        print "loading params file..."
        params = load_params_file(path_to_file)
        encodingMatrix = params['W']
        hiddenBias = params['b']            
            
        test_SdA(dataset = path_to_file, learning_rate=0.0027689, 
                training_epochs=20, n_visible=130443, n_hidden=[100],
                W=encodingMatrix, bvis=hiddenBias, activation=None,
                bhid=None)                                    
    elif not os.path.exists(get_params_file_path(path_to_file)):
        test_SdA(batch_size = MINI_BATCH_SIZE, learning_rate=0.25, \
            training_epochs=20, dataset = path_to_file, n_visible=58, \
            n_hidden=[100, 100], activation=T.nnet.sigmoid, bhid=None)
        
        #test_SdA(batch_size = MINI_BATCH_SIZE, learning_rate=0.25, \
        #    training_epochs=30, dataset = path_to_file, n_visible=58, \
        #    n_hidden=[100, 100], activation=T.nnet.sigmoid, bhid=None)
        #test_dA(dataset = path_to_file, learning_rate=0.001383, 
        #        training_epochs=20, n_visible=522483, n_hidden=100)
        #test_dA(dataset = path_to_file, learning_rate=0.0027689, 
        #        training_epochs=20,n_visible=130443, n_hidden=100)                
    else:
        asinTitleMap = obtain_key_title_mapping(asin_to_title) if asin_to_title else None
        asinIdxMap = load_asin_idx_mapping(path_to_file)
        
        print "loading params file..."
        params = load_params_file(path_to_file)

        startPos = 1295
        for doc in range(startPos, startPos + NUM_DOCS):
            most_similar(params, doc, asinTitleMap, asinIdxMap)