# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 21:11:45 2014

@author: ubuntu
"""

import os
import numpy
import math
import json

from optparse import OptionParser
from sklearn.mixture import GMM
from sklearn import cluster
from sklearn import preprocessing

import scipy
from theano import sparse

import pylab as pl
from sklearn.metrics import roc_curve, auc

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from stacked_auto_encoder_modeler import SdA

from sklearn import decomposition

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
    
def generate_svm_lite_attr_vec(idxVector):
    row = []
    col = []
    data = []
 
    def preprocess(rowId, featureVector):
        r = []
        c = []
        d = []
        for feature in featureVector:
            featureSplit = feature.split(":")
            idx, val = int(featureSplit[0]), float(featureSplit[1])
            
            r.append(rowId)
            c.append(idx)
            d.append(val)
    
        return r, c, d
            
    for idx, val in enumerate(idxVector):
        if type(val) == list:
            r, c, d = preprocess(idx, val)

            row.extend(r)
            col.extend(c)
            data.extend(d)
        else:
            raise Exception("value must be of type list")
            
    return row, col, data    

def generate_attr_vec(inputData):
    row = []
    col = []
    data = []
 
    for idx, val in enumerate(inputData):
        row.extend([idx for c in val])
        col.extend([idx for idx, c in enumerate(val)])
        data.extend([c for c in val])
                            
    return row, col, data

def generate_feature_for_csv_record(record, count):
    parsedData = map(lambda x: x.strip(), record.split(","))
    
    sample = map(float, parsedData[:len(parsedData)-1])
    label = int(parsedData[len(parsedData)-1])
    
    return sample, label


def generate_feature_for_json_record(record, count):
    jsonObject = json.loads(record)
    if "total" in jsonObject:
        return jsonObject["total"], None
    else:
        return jsonObject["x"], jsonObject["y"]

def generate_train_test_data(dataset, recordParserCallback, trainOnLabel):

    trainDataSet = []
    trainDataLabel = []
    
    testDataSet = []                   
    testDataLabel = []                       

    # create the generator for leading features                 
    feature_generator = load_feature(dataset, recordParserCallback) 
    while True:
        try:
            sample, label = feature_generator.next() 
            if label is None:
                total = sample
                continue

            trainDataSet.append(sample) if label == trainOnLabel \
                else testDataSet.append(sample)                
            trainDataLabel.append(label) if label == trainOnLabel \
                else testDataLabel.append(label)
                                
        except StopIteration:
            break

    print "# train: " + str(len(trainDataSet)) + " with label: " + str(trainOnLabel)        
    print "# test: " + str(len(testDataSet))

    #trainDataSet, trainDataLabel, validationDataSet, validationDataLabel = \
    #    generate_validation_set(trainDataSet, trainDataLabel)
    
    trainDataSet, trainDataLabel, testDataSet, testDataLabel = \
        generate_train_test_split(trainDataSet, trainDataLabel, \
            testDataSet, testDataLabel)
            
    return trainDataSet, trainDataLabel, testDataSet, testDataLabel, total
    
def runExperiment(dataset, trainOnLabel, total, learning_rate=[0.10], 
            training_epochs=15, batch_size=20, n_visible=4, n_hidden=[100], 
            modulo=100, corruption_level=[0.0], activation=None, 
            recordParserCallback=generate_feature_for_csv_record,
            attr_vec_fn=generate_attr_vec, normalize=False, params=None):

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
        

    trainDataSet, trainDataLabel, testDataSet, testDataLabel, total = \
        generate_train_test_data(dataset, recordParserCallback, trainOnLabel)

    
    
    
    def plotCurve(probabilities, testDataLabel, legend):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(testDataLabel, probabilities)
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : %f" % roc_auc)
        
        # Plot ROC curve
        #pl.clf()
        pl.plot(fpr, tpr, label=legend[-1] + ' (area = %0.2f)' % roc_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic')
        pl.legend(loc="lower right")
        #pl.show()        
        
    def plotROCCurve(fitClassifier, trainResults, trainDataLabel, 
                     testResults, testDataLabel, legend):
        probabilities = fitClassifier.score(testDataSet)
        plotCurve(probabilities, testDataLabel, legend)        
        
    pl.clf()
    legend = []
    
    #train_GMM(#validationDataSet, validationDataLabel, 
    #          trainDataSet, trainDataLabel, testDataSet, testDataLabel, 
    #          plotROCCurve, total, ["diag"], legend, normalize, attr_vec_fn)

    # train the K-Means
    def processOutputKMeans(trainDataSet, trainDataLabel, testDataSet, 
                testDataLabel, closeCenterDistTrain, closeCenterDistTest,
                legend):
                
        plotCurve(closeCenterDistTest, testDataLabel, legend)        
               
    print "training k-means..."
    #train_KMeans(trainDataSet, trainDataLabel, testDataSet, testDataLabel, 
    #              processOutputKMeans, legend, total, "", normalize, 
    #              attr_vec_fn)    

    # train the auto-encoder
    def processOutputs(classifier, trainDataSet, trainDataLabel, 
                       testDataSet, testDataLabel, generate_attr_vec_callback, 
                       legend, total):
        def generate_errors(dataSet):
            testMatrix = generate_feature(dataSet, total, generate_attr_vec_callback)
            errorVectorTest = classifier.get_reconstruction_errors(testMatrix.tocsc())    
    
            error = errorVectorTest
            sqrdErrorVector = numpy.diag(numpy.dot(error, numpy.transpose(error)))

            return sqrdErrorVector

        probabilities = generate_errors(testDataSet)            
        plotCurve(probabilities, testDataLabel, legend)        

    n_hid = []                           
    n_corruption_levels = []
    n_learning_rates = []
    if params is not None:
        n_hidden = params["n_hidden"]
        learning_rate = params['learning_rates']
        corruption_level = params['corruption_levels']
                        
    for i, n in enumerate(n_hidden):
        n_hid.append(n)
        n_corruption_levels.append(corruption_level[i])
        n_learning_rates.append(learning_rate[i])
        
        transformSample, transformTest = train_autoEncoder(trainDataSet, 
            testDataSet, trainDataLabel, testDataLabel, n_visible, n_hid, 
            n_learning_rates, processOutputs, n_corruption_levels, 
            activation, training_epochs, legend, normalize, attr_vec_fn, 
            total, i, params)    
                 
        train_KMeans(transformSample, trainDataLabel, transformTest, testDataLabel, 
                      processOutputKMeans, legend, n,
                      " transformed " + str(i) + " layer", True, 
                      generate_attr_vec)
        
    
    #train_autoEncoder(trainDataSet, testDataSet, trainDataLabel, 
    #    testDataLabel, n_visible, n_hidden, learning_rate, processOutputs, 
    #    corruption_level, activation, training_epochs)        

    pl.show()
    plot_transformed_vectors(transformTest, testDataLabel, title="after feature learning " + str(i) + " layer")
    
    
def plot_transformed_vectors(transformedFeatures, dataLabels, colors=['b','r'], title=""):

    #scaler = preprocessing.Scaler().fit(transformedFeatures)
    #trainData = scaler.transform(transformedFeatures)

    labels = numpy.unique(dataLabels)    
    if len(colors) != len(labels):
        raise Exception("need color array to be same size as unique labels")
        
    pca = decomposition.PCA(n_components=2, whiten=True)
    pca.fit(transformedFeatures)

    new_features = pca.transform(transformedFeatures)    

    x_features_to_plot = []
    y_features_to_plot = []
    
    pl.clf()    
    for label, color in zip(labels, colors):
        for idx, feature in enumerate(new_features):
            if dataLabels[idx] == label:            
                x_features_to_plot.append(feature[0]) 
                y_features_to_plot.append(feature[1])
                
        pl.scatter(x_features_to_plot, y_features_to_plot, c=color)
        pl.title(title)
        
        x_features_to_plot = []
        y_features_to_plot = []
        
    pl.show()
        
    
def get_params_file(feature_file):        
    return os.path.join(DATA_DIR_PATH, os.path.basename(
        feature_file).split(".")[0] + "_params.out")
    
def generate_feature(inputData, total, callback):

    rowL, colL, dataL = callback(inputData)         
    sampleMatrix = scipy.sparse.coo_matrix( (dataL, (rowL, colL)), \
        shape=(len(inputData), total), dtype=numpy.float32)
    
    return sampleMatrix
    
def load_params_file(feature_file):
    """
        Function to load the learned parameters to memory.
    """
    
    params_file_path = get_params_file(feature_file)
    
    infile = numpy.load(params_file_path)
    return infile
    
def train_autoEncoder(trainData, testData, trainDataLabel, testDataLabel, 
                      n_visible, n_hidden, learning_rates, callback, 
                      corruption_levels, activation, training_epochs, legend,
                      normalize, generate_attr_vec_callback, total, layer_no, 
                      params=None):

    if len(corruption_levels) != len(n_hidden):
        raise Exception("corruption level not provided for each layer...will use default")

    if len(learning_rates) != len(n_hidden):
        raise Exception("learning rates not provided for each layer...will use default")
        
    legend.append("SAE %d layers" % len(n_hidden))    
    
    sample = sparse.csc_matrix(name='s', dtype='float32') 
    label = sparse.csc_matrix(name='l', dtype='float32')      
    x = sparse.csc_matrix(name='x', dtype='float32')  
    y = sparse.csc_matrix(name='y', dtype='float32')  

    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    #bhid_value = numpy.zeros(n_hidden, dtype=theano.config.floatX)                           
    W = None
    b = None
    if params is not None:    
        W = params['W']
        if len(params['b']) == len(params['W']):
            b = params['b']
                    
    sda = SdA(numpy_rng=rng, theano_rng=theano_rng, n_ins=n_visible,
              hidden_layers_sizes=n_hidden, input=x, label=y,
              activation=activation, W=W, b=b)
                  #activation=activation, b=bhid_value)
        
    '''cost, updates = sda.get_cost_updates(corruption_level=corruption_level,
                                        learning_rate=learning_rate)

    # this is the function that theano will call to optimize the cost
    # function.
    train_da = theano.function([sample, label], cost, updates=updates,
         givens={x: sample,
                 y: label})
    '''
    
    pretraining_fns = sda.pretraining_functions(sample, label, learning_rate, 
                                                corruption_level)
    
    if normalize:
        def normalize_data(data):
            # normalize data
            inputArray = numpy.array(data)
            
            minValue = inputArray.flatten().min()
            maxValue = inputArray.flatten().max()    
            inputArray = (inputArray - float(minValue))/(float(maxValue - minValue))
            
            return inputArray
            
        trainData = normalize_data(trainData)
        testData = normalize_data(testData)
        
    #sampleMatrix = generate_feature(trainData, total, generate_attr_vec_callback)
    for idx, fn in enumerate(pretraining_fns):
        print "training layer #%s" % str(idx)                
        for i in range(0, training_epochs):
            for sample in trainData:
                sampleMatrix = generate_feature([sample], total, generate_attr_vec_callback)
                error = fn(sampleMatrix, sampleMatrix, 
                           corruption_levels[idx], learning_rates[idx])
            print "error train cost: " + str(error)

    sampleMatrix = generate_feature(trainData, total, generate_attr_vec_callback)
    errorVector = sda.get_reconstruction_errors(sampleMatrix.tocsc())   

    testMatrix = generate_feature(testData, total, generate_attr_vec_callback)
    errorVectorTest = sda.get_reconstruction_errors(testMatrix.tocsc())    
    
    # plot without before feature learning
    plot_transformed_vectors(testMatrix.toarray(), testDataLabel, title="before feature learning")
    

    def find_avg_error(errorMatrix):
        error = errorMatrix
        sqrdErrorMatrix = numpy.dot(error, numpy.transpose(error))
        return numpy.diag(sqrdErrorMatrix)

    print "error train: " + str(math.sqrt(sum(find_avg_error(errorVector))))
    print "error test: " + str(math.sqrt(sum(find_avg_error(errorVectorTest))))
    
    # look at individual errors:
    callback(sda, trainData, trainDataLabel, testData, 
             testDataLabel, generate_attr_vec_callback, legend, total)

    transformSample = sda.get_hidden_values(sampleMatrix.tocsc())
    transformTest = sda.get_hidden_values(testMatrix.tocsc())
    
    return transformSample.eval(), transformTest.eval()

def k_means_metric1(dataSet, centers):
    closeCenterDist = []
    numRows = dataSet.shape[0]
    for row in range(0, numRows):
        sample = dataSet.getrow(row).toarray()[0]
        minDistance = math.pow(10,100)            
        for centerId, center in enumerate(centers):
            distance = sum((sample-center)**2)
            #distance = [(si-ci) ** 2 for si, ci in zip(sample,center)]
            #distance = math.sqrt(numpy.dot(sample-center, sample-center))
            distance = math.sqrt(distance)
            if distance < minDistance:
                minDistance = distance
        
        closeCenterDist.append(minDistance)
    return closeCenterDist

def k_means_metric2(dataSet, centers):
    centerDist = []
    numRows = dataSet.shape[0]
    for row in range(0, numRows):
        sample = dataSet.getrow(row).toarray()[0]
        
        distanceSum = 0
        for centerId, center in enumerate(centers):
            squaredDistance = sum((sample-center)**2)
            #squaredDistance = [(si-ci) ** 2 for si, ci in zip(sample,center)]
            distance = math.sqrt(squaredDistance)
            distanceSum = distanceSum + distance
                
        avgDistance = distanceSum / len(centers)
        centerDist.append(avgDistance)
    return centerDist
    
def train_KMeans(trainDataSet, trainDataLabel, testDataSet, testDataLabel, 
                  callback, legend, total, suffix="", normalize=False, 
                  attr_vec_fn=generate_attr_vec):

    trainData = trainDataSet
    testData = testDataSet

    if normalize:
        scaler = preprocessing.Scaler().fit(trainDataSet)
        trainData = scaler.transform(trainDataSet)
        testData = scaler.transform(testDataSet)

    trainData = generate_feature(trainData, total, attr_vec_fn)
    testData = generate_feature(testData, total, attr_vec_fn)

    
    k_means = cluster.KMeans(k=90)
    k_means.fit(trainData) 
    
    centers = k_means.cluster_centers_    
    
    # metric #1: distance to nearest centroid
    legend.append("K-Means #1" + suffix)
    
    closeCenterDistTrain = k_means_metric1(trainData, centers)
    closeCenterDistTest = k_means_metric1(testData, centers)    
    callback(trainData, trainDataLabel, testData, testDataLabel, 
             closeCenterDistTrain, closeCenterDistTest, legend)
        
    # metric #2: distance from avg distance to all centroids    
    legend.append("K-Means #2" + suffix)    
        
    closeCenterDistTrain = k_means_metric2(trainData, centers)
    closeCenterDistTest = k_means_metric2(testData, centers)        
    callback(trainData, trainDataLabel, testData, testDataLabel, 
             closeCenterDistTrain, closeCenterDistTest, legend)        
        
def train_GMM(#validationDataSet, validationDataLabel, 
              trainDataSet, trainDataLabel, testDataSet, testDataLabel, 
                  callback, total, covars=['spherical', 'diag', 'tied', 'full'],
                  legend=[], normalize=False, attr_vec_fn=generate_attr_vec):

    legend.append("GMM")
    
    trainData = trainDataSet
    testData = testDataSet
    if normalize:
        scaler = preprocessing.Scaler().fit(trainDataSet)
        #validDataSet = scaler.transform(validationDataSet)
        trainData = scaler.transform(trainDataSet)
        testData = scaler.transform(testDataSet)
        
    def get_components():
        pass

    trainData = generate_feature(trainData, total, attr_vec_fn)
    testData = generate_feature(testData, total, attr_vec_fn)
    
    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=2,
        covariance_type=covar_type, init_params='wc', n_iter=40))
                for covar_type in covars)
                   
    for index, (name, classifier) in enumerate(classifiers.iteritems()):        
        # Train the other parameters using the EM algorithm.
        fitClassifier = classifier.fit(trainData)
        
        y_train_pred = classifier.score(trainData)
        y_test_pred = classifier.score(testData)
        
        callback(fitClassifier, y_train_pred, trainDataLabel, y_test_pred, 
                 testDataLabel, legend)                               

def generate_validation_set(trainDataSet, trainDataLabel):

    samplesToIncludeInTesting = int(len(trainDataSet)*.5) 
    print "removing %s samples from training set and adding to test set" \
        % str(samplesToIncludeInTesting)
    
    validationDataSet = []
    validationDataLabels = []
    for sampleCount in range(0, samplesToIncludeInTesting):
        sample = trainDataSet.pop(sampleCount)
        label = trainDataLabel.pop(sampleCount)
        
        validationDataSet.append(sample)
        validationDataLabels.append(label)
        
    return trainDataSet, trainDataLabel, validationDataSet, validationDataLabels
    
def generate_train_test_split(trainDataSet, trainDataLabel, 
                              testDataSet, testDataLabel):

    samplesToIncludeInTesting = int(len(trainDataSet)*.25) 
    print "removing %s samples from training set and adding to test set" \
        % str(samplesToIncludeInTesting)
    
    for sampleCount in range(0, samplesToIncludeInTesting):
        sample = trainDataSet.pop(sampleCount)
        label = trainDataLabel.pop(sampleCount)
        
        testDataSet.append(sample)
        testDataLabel.append(label)
        
    return trainDataSet, trainDataLabel, testDataSet, testDataLabel

    
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-u", "--update", action = "store_true", dest="update",
                      help="update a given model")
    parser.add_option("-l", "--trainLabel", dest="label", type="int", 
                      action="store", help="train label value")                      
    parser.add_option("-f", "--file", dest="file", action="store",
                      help="path to training data", metavar="FILE")
    parser.add_option("-i", "--initialize", dest="params", action="store_true",
                      help="initialize net params")
    
    (options, args) = parser.parse_args()

    if options.label:
        label = int(options.label)
    else:
        label = 0
        
    if options.file:
        fileName = options.file
    else:
        fileName = "../data/data_banknote_authentication.txt"

    DATA_DIR_PATH = os.path.dirname(fileName)    
        
    if options.params:
        params = load_params_file(fileName)
    else:
        params = None
        
    #runExperiment(fileName, label, learning_rate=0.10, training_epochs=30, 
    #        n_visible=4, n_hidden=10, modulo=100, 
    #        corruption_level=0.0, activation=T.tanh)
    #runExperiment(fileName, label, total=4, learning_rate=[0.10, 0.10],  
    #        training_epochs=20, n_visible=4, n_hidden=[100, 100], modulo=100, 
    #        corruption_level=[0.9, 0.8], activation=T.nnet.sigmoid,
    #        recordParserCallback=generate_feature_for_csv_record,
    #        attr_vec_fn=generate_attr_vec, normalize=True)
    runExperiment(fileName, label, total=64, learning_rate=[.25], 
            training_epochs=1, n_visible=64, n_hidden=[10], 
            modulo=100, corruption_level=[0.0], activation=T.nnet.sigmoid,    
            recordParserCallback=generate_feature_for_json_record,
            attr_vec_fn=generate_svm_lite_attr_vec, normalize=False, 
            params=params)