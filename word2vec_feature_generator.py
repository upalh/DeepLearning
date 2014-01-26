"""
This script creates the feature vectors that will be fed into word2vec.
Basically all its doing is getting the list of customers that reviewed 
each product and the list of products that are similar to it. And then
it will write out the pair <customerId, productASIN>. So basially, we're
assuming that everyone that's written a review for the current product
also bought all the similar products too.

The script can also be used to create feature vectors in the opposite 
direction. That is for each product, map it to all the customers that have
bought it. 

Example format:
    "key_A2YD21XOPJ966C"       "['0790747324', '6305350221']"
    
Example command:
    python word2vec_feature_generator.py -r local --primary_key reviews 
        --value_key similar --meta_key_key ASIN --primary_index 0 
            ../data/amazon/amazon-meta_preprocessed.out > features_for_word2vec

    python word2vec_feature_generator.py -r local --primary_key similar 
        --value_key reviews --meta_key_key ASIN --value_index 0 
            ../data/amazon/test.out > word2vec_features_value_to_key
            
    python word2vec_feature_generator.py -r local --primary_key ASIN 
        --value_key similar ../data/amazon/test.out > word2vec_features_value_to_key            

Created on Thu Jan 23 06:57:00 2014

@author: Upal Hasan
"""
import sys
import json
from mrjob.job import MRJob

DEFAULT_KEY_MAPPING = "key"
    
class FeatureGenerator(MRJob):

    def configure_options(self):
        super(FeatureGenerator, self).configure_options()
        self.add_passthrough_option(
            '--primary_key', help="primary key to perform the mapping over.")
        self.add_passthrough_option(
            '--primary_index', help="if primary key is list, which index should be the key.")                        
        self.add_passthrough_option(
            '--meta_key_key', help="key for a metadata to be included in the mapping to primary_key list.")           
        self.add_passthrough_option(
            '--meta_key_value', help="key for a metadata to be included in the mapping to value_key list.")                       
        self.add_passthrough_option(
            '--value_key', help="value key that primary_key will use to map to.")                    
        self.add_passthrough_option(
            '--value_index', help="if value key is list, which index should be the value.")                        
            
                            
    def mapper(self, _, line):
        metaKeyKey = self.options.meta_key_key
        metaKeyValue = self.options.meta_key_value                
        primaryIdx = int(self.options.primary_index) if self.options.primary_index else None
        primaryKey = self.options.primary_key
        valueKey = self.options.value_key
        valueIdx = int(self.options.value_index) if self.options.value_index else None
        
        parsedLine = json.loads(line)        
        
        if primaryKey in parsedLine and valueKey in parsedLine:
                   
            primaryKeyElement = parsedLine[primaryKey]
            keys =  primaryKeyElement if type(primaryKeyElement) == list else [primaryKeyElement]
            values = parsedLine[valueKey]

            # include the metadata if there are any
            if metaKeyKey in parsedLine or metaKeyValue in parsedLine: 
                metaKey = metaKeyKey if metaKeyKey else metaKeyValue                            
                meta = parsedLine[metaKey]

                if metaKeyKey: 
                    keys.append(meta) 
                else:
                    values.append(meta)

            def extract_key_value(key, keyIdx, prefix):        
                if type(key) == list:
                    if type(keyIdx) != int:
                        raise Exception("Id index for key must be a valid integer for key " + str(key))
                    keyElement = key[keyIdx]
                else:
                    keyElement = key
                    
                return prefix + keyElement

            for key in keys:
                for value in values:            
                    writableKey = extract_key_value(key, primaryIdx, "key_")                    
                    writableValue = extract_key_value(value, valueIdx, "")                    
                                        
                    yield writableKey, writableValue
            
    def reducer(self, key, values):
        yield key, str(list(values))
        
if __name__ == '__main__':
    FeatureGenerator.run()