# -*- coding: utf-8 -*-
"""
Class to perform some analysis on the amazon dataset using mrjob.

Example command:
    python amazon_data_analyzer.py data/amazon/amazon-meta_preprocessed.out 
        -r local --customer reviews --similar similar --product ASIN 
            --customer_index 0

Created on Sun Jan 19 05:46:37 2014

@author: mhasan

"""
import sys
import json
from mrjob.job import MRJob

class AmazonDataAnalyzer(MRJob):

    def configure_options(self):
        super(AmazonDataAnalyzer, self).configure_options()
        self.add_passthrough_option(
            '--customer', help="customer key")
        self.add_passthrough_option(
            '--customer_index', help="customer index within customer array.")                        
        self.add_passthrough_option(
            '--product', help="product key (ASIN).")
        self.add_passthrough_option(
            '--similar', help="similar product key.")            
        self.add_passthrough_option(
            '--metric', help="metric for evaluation: '1' for products that " \
                                "have no customers. '2' for products that " \
                                "are discontinued. '3' for number of customers " \
                                "that have bought just 1 product. '4' for " \
                                "distribution of # product purchases.")                        
    
        
    def mapper(self, _, line):
        productKey = self.options.product
        customerKey = self.options.customer
        customerIdx = int(self.options.customer_index)        
        similarKey = self.options.similar
        metric = int(self.options.metric)
        
        parsedLine = json.loads(line)        
        
        if metric == 1:
            # how many products have no customers ==> "no_customers"	139960
            if customerKey in parsedLine and not parsedLine[customerKey]:
                yield "no_customers", 1

        elif metric == 2:
            # how many discontinued products ==> "discontinued_products" 5868
            if customerKey not in parsedLine and similarKey not in parsedLine:
                yield "discontinued_products", 1

        elif metric == 3:
            # how many customers bought just one product
            # in the reducer we will have to filter all the customer IDs
            # whose count is greater than 1.        
            # there appear to be 1555170 customer entries, but 744954 have only
            #   bought 1 product
            customerList = parsedLine[customerKey] if customerKey in parsedLine else None
            if customerList:
                for customer in customerList:
                    yield "customer_" + customer[customerIdx], 1
        
        else:            
            # distribution of # product purchases
            # there are 721342 unique ASINS, but a total of 958609 ASINS, 
            # and 548552 records, so only about 200k of the actual records were
            # in the similar sections. the others were new that don't have a 
            # corresponding record for it.
            if productKey in parsedLine:        
                yield "product_" + parsedLine[productKey], 1
                
            similarList = parsedLine[similarKey] if similarKey in parsedLine else None                
            if similarList:
                for similarProduct in similarList:
                    yield "product_" + similarProduct, 1
            
        
    def reducer(self, key, values):
        if "customer_" in key:
            valueSums = sum(values)
            if valueSums == 1:
                yield key, valueSums
        else:
            yield key, sum(values)


if __name__ == '__main__':
    AmazonDataAnalyzer.run()
    

