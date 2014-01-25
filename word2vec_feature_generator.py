"""
This script creates the feature vectors that will be fed into word2vec.
Basically all its doing is getting the list of customers that reviewed 
each product and the list of products that are similar to it. And then
it will write out the pair <customerId, productASIN>. So basially, we're
assuming that everyone that's written a review for the current product
also bought all the similar products too.

Example format:
    "customer_A2YD21XOPJ966C"       "['0790747324', '6305350221']"
    
Example command:
    python word2vec_feature_generator.py -r local --customer reviews 
        --similar similar --product ASIN --customer_index 0 
            ../data/amazon/amazon-meta_preprocessed.out > features_for_word2vec


Created on Thu Jan 23 06:57:00 2014

@author: Upal Hasan
"""

import json
from mrjob.job import MRJob

class FeatureGenerator(MRJob):

    def configure_options(self):
        super(FeatureGenerator, self).configure_options()
        self.add_passthrough_option(
            '--customer', help="customer key")
        self.add_passthrough_option(
            '--customer_index', help="customer index within customer array.")                        
        self.add_passthrough_option(
            '--product', help="product key (ASIN).")           
        self.add_passthrough_option(
            '--similar', help="similar product key.")            
    
        
    def mapper(self, _, line):
        productKey = self.options.product
        customerKey = self.options.customer
        customerIdx = int(self.options.customer_index)        
        similarKey = self.options.similar
        
        parsedLine = json.loads(line)        
        
        if customerKey in parsedLine and productKey in parsedLine \
            and similarKey in parsedLine:
                            
            customerList = parsedLine[customerKey]
            product = parsedLine[productKey]
            similarItems = parsedLine[similarKey]
            
            similarItems.append(product)
            if customerList:
                for customer in customerList:
                    for item in similarItems:
                        yield "customer_" + customer[customerIdx], item                  
        
    def reducer(self, key, values):
        yield key, str(list(values))

class FeatureMetadataGenerator(MRJob):

    def configure_options(self):
        super(FeatureGenerator, self).configure_options()
        self.add_passthrough_option(
            '--product', help="product key (ASIN).")
        self.add_passthrough_option(
            '--title', help="title key.")            
        
    def mapper(self, _, line):
        productKey = self.options.product
        titleKey = self.options.title
        
        parsedLine = json.loads(line)        
        
        if productKey in parsedLine and titleKey in parsedLine:                            
            product = parsedLine[productKey]
            title = parsedLine[titleKey]
            
            yield product, title
        
    def reducer(self, key, values):
        yield key, str(list(values))
        
if __name__ == '__main__':
    FeatureGenerator.run()