"""
This script will load the feature vectors that were generated by 
word2vec_feature_generator.py and then train the net to learn a more 
distributed representation of each word. In the case of the amazon dataset,
each word corresponds to a product that a customer bought. So a sentence
would correspond to all the products that a given customer bought.

This script should be usable for any situation where the learned feature 
vectors have been created with word2vec_feature_generator.py or are
similarly formatted.

Example format of existing feature file should be:
    "customer_A2YD21XOPJ966C"       "['0790747324', '6305350221']"

TODO:
    Assumes that all the sentences will fit into memory before 
    feeding into Word2Vec. Might be a good idea to create a 
    generator to reduce memory issues.


Created on Sat Jan 25 05:44:22 2014

@author: Upal Hasan

"""

import sys, os, time
import numpy
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import Counter
from sklearn import cluster


DEFAULT_BASE_DIR = ""
    
def load_features(path_to_file, callback):
    """This function will load the features, parse them out, and return
        to the caller for processing.
        
        Args: 
            path_to_file : path to the feature vectors file
            
        Return:
            An array corresponding to the corresponding to a parsed record.        
    """
    
    with open(path_to_file, "r") as fileHandle:
        while True:
            record = fileHandle.readline()
            if not record:
                break
            
            # call the callback function to retrieve the value of interest 
            # to return to the caller.             
            value = callback(record)
        
            yield value


def plot_learned_vectors2D(wgtMatrixPath, model, dim):
    """Function to help us understand what was learned by plotting
        the new representations of the words in the sentences.
        
        Args:
            wgtMatrixPath : path to the weight matrix path of learned vectors.
            model : the model learned by word2vec
    """
    with open(wgtMatrixPath, "r") as fileHandle:        
        vectorList = map(lambda x : x.rstrip().split(" ")[0], fileHandle.readlines())
        
    words = vectorList[:200]
    
    data, annotations = get_sentence_data(wgtMatrixPath, model, dim, words)
    xs = [element[0] for element in data]        
    ys = [element[1] for element in data]
    
    fig, ax = plt.subplots()
            
    ax.scatter(xs, ys)
    
    # annotate the chart
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (xs[i], ys[i]))
    plt.show()

def preprocess_record(line):
    """Function to parse a record that was produced by 
        word2vec_feature_generator.py. It returns the corresponding 
        key and value.
        
        Args:
            line : The line of record that we're processing.
            
        Returns:
            An array with the parsed elemets.
    """
    record = line.replace("'","")
    record = record.replace("\"","")            
    
    val = record.strip()
    val = val.replace("[","")
    val = val.replace("]","")

    key, value = val.split("\t")                
    valSplit = value.split(", ")                    
    
    return key, valSplit

def get_sentence_data(wgtMatrixPath, model, dim, words):
    """Function to extract out the feature vector dimension for a 
        given sentence vector.
        
        Args:
            wgtMatrixPath : path to the weight matrix path of learned vectors.
            model : the model learned by word2vec
            dim : the diension of the learned vectors
            words : the set of wrods in a sentence

        Return:
            Returns the data vector which is a vector of dim vectors. So each
            index corresponds to each dim dimension.
    """
    
    with open(wgtMatrixPath, "r") as fileHandle:        
        vectorDim = fileHandle.readline().rstrip().split(" ")[1]
            
    if int(vectorDim) != dim:
        raise Exception("vector list empty or not proper dimension.")

    # extract out the words from the weight matrix file.
    #data = [[] for i in range(0, dim)]  
    data = []       
    annotations = []
    for word in words:
        try:        
            w = model[word]
            data.append(w)
            #for i in range(0, len(w)):
            #    data[i].append(w[i])                
            annotations.append(word)
        except:
            continue
            
    return data, annotations

def plot_entire_feature_vectors3D(model, path_to_file):
    """Function to help us understand what was learned by plotting
        the new representations of the sentences, which now is a word of
        words. Plot a 3D representation of it.
        
        Args:
            model : the model learned by word2vec
            path_to_file : the path to the file that contains the features to
                be fed to word2vec.
            
    """
    num_sentences_to_plot = 1000
    num_dimensions = 3
    data, annotations = get_sentences_data(path_to_file, num_sentences_to_plot, 
                                          num_dimensions, model)
    xs = [element[0] for element in data]        
    ys = [element[1] for element in data]
    zs = [element[2] for element in data]

    fig = plt.figure()
    ax = Axes3D(fig)
              
    ax.scatter(xs, ys, zs, c=range(0, len(xs)))    
    plt.show()

def get_sentences_data(path_to_file, num_sentences_to_plot, dim, model):
    """Function to get extracted vector representations for all the 
        sentences. For each sentence it calls get_sentence_data to
        get the data for each sentence and then appends them together.

        Args:
            path_to_file : path to the file representing features to be passed
                into word2vec.
            num_sentences_to_plot : the number of sentences that we will plot,
                where each sentence is a vector of vectors representation for
                a given line in path_to_file.
            dim : the number of dimensions for each leanred vector.
            model : the model learned by word2vec


        Return:
            Returns the data vector which is a vector of dim vectors. So each
            index corresponds to each dim dimension.
    """
    
    keys = []
    sentences = []
    featureGenerator = load_features(path_to_file, preprocess_record)    
    for i in range(0, num_sentences_to_plot):
        key, value = featureGenerator.next()                 
        
        keys.append(key)
        sentences.append(value)

    wgtMatrixPath = get_weight_matrix_path(path_to_file)
    
    #data = [[] for i in range(0, dim)]
    data = []
    annotations = []
    for i, sentence in enumerate(sentences):
        sentenceData, sentenceAnnotations = get_sentence_data(wgtMatrixPath,
            model, dim, sentence)

        data.extend(sentenceData)        
        #for j in range(0,dim):
        #    data[j].extend(sentenceData[j])
        annotations.extend(sentenceAnnotations)

    return data, annotations
    
def plot_entire_feature_vectors2D(model, path_to_file):
    """Function to help us understand what was learned by plotting
        the new representations of the sentences, which now is a word of
        words. Plot a 2D version of it.
        
        Args:
            model : the model learned by word2vec
            path_to_file : the path to the file that contains the features to
                be fed to word2vec.
            
    """
    num_dimensions = 2
    num_sentences_to_plot = 500
    
    data, annotations = get_sentences_data(path_to_file, num_sentences_to_plot, 
                                          num_dimensions, model)

    xs = [element[0] for element in data]        
    ys = [element[1] for element in data]
        
    fig, ax = plt.subplots()
        
    ax.scatter(xs, ys, s=100, c=range(0,len(xs)))
    
    # annotate the chart
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (xs[i], ys[i]))
    plt.show()
    
    
def get_model_path(path_to_feature_file):
    """Function to obtain the path to the model file.
    
        Args:
            path_to_feature_file : path to the file containing all the
                features to be fed into word2vec.
        Returns:
            The path where the model file will/should reside.
    """
    return os.path.join(DEFAULT_BASE_DIR, os.path.basename(path_to_feature_file).split(".")[0] + "_model.out")    

def get_weight_matrix_path(path_to_feature_file):
    """Function to obtain the path to the weight matrix file.
    
        Args:
            path_to_feature_file : path to the file containing all the
                features to be fed into word2vec.
        Returns:
            The path where the weight matrix will/should reside.
    """    
    return os.path.join(DEFAULT_BASE_DIR, os.path.basename(path_to_feature_file).split(".")[0] + "_weight.out")    

def get_asin_to_title_path(path_to_json_meta):
    return os.path.join(DEFAULT_BASE_DIR, "ASIN_to_titles")        

def reduce_dimension_and_plot(wgtMatrixPath):
    
    with open(wgtMatrixPath, "r") as fileHandle:        
        vectorDim = fileHandle.readline().rstrip().split(" ")[1]
        
    num_dimensions = 2
    num_sentences_to_plot = 2
    
    data, annotations = get_sentences_data(path_to_file, num_sentences_to_plot, 
                                          int(vectorDim), model)
    data = numpy.array(data)                                

    # do a dimensionality reduction if vectors are of high dimensionality    
    pca = PCA(n_components=num_dimensions, whiten=True)
    data_prime = pca.fit(data).transform(data)
    
    xs = [element[0] for element in data_prime]        
    ys = [element[1] for element in data_prime]    

    fig, ax = plt.subplots()
            
    ax.scatter(xs, ys)
        
    # annotate the chart
    for i, txt in enumerate(annotations):
        ax.annotate(txt, (xs[i], ys[i]))
    plt.show()


def cluster_sentence_vectors(model, path_to_file, dim):
    
    num_sentences_to_cluster = 2
    num_dimensions = dim
    
    data, annotations = get_sentences_data(path_to_file, num_sentences_to_cluster, 
                                          num_dimensions, model)

   # data = numpy.array(data)
    
    print "data: " + str(annotations)
    labels = []
    for i in range(0, 100):
        k_means = cluster.KMeans(k=num_sentences_to_cluster)
        k_means.fit(data) 
    
        labels.append(k_means.labels_)

    final_labels = []
    for i in range(0, len(annotations)):
        idx_labels = [label[i] for label in labels]
        freqs = Counter(idx_labels)
        final_labels.append(freqs.most_common(1)[0][0])

    print "clusters: " + str(final_labels)
    
def train_model(path_to_file, size, window, min_count):
    """Function to train thd deep learning model.
    
        Args:
            path_to_file : path to the file that contains features to feed 
                into Word2Vec.
            size : size of the learned vectors for Word2Vec
            window : size of window for Word2Vec
            
        Returns:
            A learned model by Word2Vec.
    """
    
    print "training model..."        
    time_start = time.time()
    
    # helper function to help preprocess a piece of text
    def preprocess(line):
        key, value = preprocess_record(line)        
        return value
    
    model = Word2Vec(load_features(path_to_file, preprocess), size=size, window=window, min_count=min_count, workers=4)
    model.save(get_model_path(path_to_file))
    model.save_word2vec_format(get_weight_matrix_path(path_to_file))    
    time_end = time.time()
    
    time_diff = time_end - time_start
    print "saved trained model..."
    print "time: " + str(time_diff/60.0) + " mins"

    return model

def print_simliar_words(asinTitlePath, wgtMatrixPath, model):

    products = dict()
    for line in open(asinTitlePath, "r"):       
        line = line.replace('"',"")
        key, value = line.rstrip().split("\t")
        products[key] = value
        
    with open(wgtMatrixPath, "r") as fileHandle:        
        vectorDim = fileHandle.readline()
        vectorList = map(lambda x : x.rstrip().split(" ")[0], fileHandle.readlines())

    words = vectorList[:10000]
    for word in words:
        if word not in model or word not in products:
            continue

        print "processing: " + word + " " + "title: " + products[word]
        similar_words = model.most_similar(positive=[word], topn=5)
        for similar in similar_words:
            if similar[0] not in products:
                continue
            print "ASIN: " + similar[0] + " " + "title: " + products[similar[0]]
        print "=============================================="
        
    
if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        path_to_file = sys.argv[1]
    else:
        path_to_file = "../data/amazon/features_for_word2vec"

    DEFAULT_BASE_DIR = os.path.dirname(path_to_file)    
    #keys, features = load_features(path_to_file)    

    if os.path.exists(get_weight_matrix_path(path_to_file)):
        print "loading existing model..."
        model = Word2Vec.load_word2vec_format(get_weight_matrix_path(path_to_file), binary=False)
    else:   
        # now dump into word2vec
        model = train_model(path_to_file, 500, 5, 5)
     
    print_simliar_words(get_asin_to_title_path(path_to_file), get_weight_matrix_path(path_to_file), model)     
    #cluster_sentence_vectors(model, path_to_file, 3) 
    #reduce_dimension_and_plot(get_weight_matrix_path(path_to_file))
    #plot_learned_vectors2D(get_weight_matrix_path(path_to_file), model, 2)
    #plot_entire_feature_vectors2D(model, path_to_file)
    #plot_entire_feature_vectors3D(model, path_to_file)