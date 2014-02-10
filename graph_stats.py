# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:26:37 2014

@author: ubuntu
"""
import sys
import networkx as nx
import json
import os

        
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

def save_records(path, records, modulo = 1000):
    count = 0    
    with open(path, "w") as fileHandle:
        for record in records:
            json.dump(record, fileHandle)
            fileHandle.write("\n")
            if count % modulo == 1:
                print "wrote record: %s" % str(count)
            count = count + 1        

def get_save_path(path_to_feature_file):
    return os.path.join(DEFAULT_BASE_DIR, os.path.basename(path_to_feature_file).split(".")[0] + "_largest_component.out")    
    
if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        path_to_file = sys.argv[1]
    else:
        path_to_file = "../data/amazon/word2vec_features_value_to_key"

    DEFAULT_BASE_DIR = os.path.dirname(path_to_file)        
    
    G = nx.Graph()    
    for adjList in open(path_to_file):    
        node, adjacentNodes = preprocess_record(adjList)
        for adjacentNode in adjacentNodes:
            G.add_edge(node, adjacentNode)

    components = nx.connected_component_subgraphs(G)
    #print "Number of connected components: " + str(len(components))
    #print nx.number_of_nodes(components[0])

    records = []    
    biggestSubGraph = components[0]
    for node in biggestSubGraph.adj.keys():
        nodeDict = dict()
        adjacent = biggestSubGraph.adj[node].keys()

        nodeDict["ASIN"] = node
        nodeDict["similar"] = adjacent

        records.append(nodeDict)
        #print "\"" + node + "\"" + "\t"+str(adjacent)
    
    save_records(get_save_path(path_to_file), records)