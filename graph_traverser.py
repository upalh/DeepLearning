# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:40:21 2014

@author: ubuntu
"""

import sys
import json

def obtain_key_title_mapping(asinTitlePath):
    
    products = dict()
    for line in open(asinTitlePath, "r"):       
        line = line.replace('"',"")
        key, value = line.rstrip().split("\t")
        products[key] = value

    return products

if __name__ == "__main__":
    
    if(len(sys.argv)>2):
        path_to_file = sys.argv[1]
        path_to_asin_title_file = sys.argv[2]
    else:
        print "need to provide path to graph and asin_to_title mapping"
        
    asinTitleDict = obtain_key_title_mapping(path_to_asin_title_file)                
    records = dict()
    for line in open(path_to_file):
        record = json.loads(line)
        if "ASIN" in record and record["ASIN"] in asinTitleDict and "similar" in record:
            dataList = [record["ASIN"], asinTitleDict[record["ASIN"]], record["similar"]]
            records[record["ASIN"]] = dataList
    
    while True:
        asin = raw_input("Enter ASIN: ")  
        asin = asin.strip()
        if asin in records:
            print str(records[asin])
