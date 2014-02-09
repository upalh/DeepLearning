# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 01:33:19 2014

@author: ubuntu
"""

import sys
import networkx as nx
import matplotlib.pyplot as plt

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
        
if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        path_to_file = sys.argv[1]
    else:
        path_to_file = "../data/amazon/word2vec_features_value_to_key"

    GRAPH_SIZE = 500
    G = nx.Graph()
    
    count = 0
    for adjList in open(path_to_file):    
        node, edges = preprocess_record(adjList)

        G.add_node(node)
        for edge in edges:
            G.add_edge(node, edge)
        
        count = count + 1
        if count == GRAPH_SIZE:
            break
        
    print "done initializing..."
    plt.figure(figsize = (8,8))
    nx.draw_networkx(G, pos = nx.spring_layout(G))
    
    '''
    nx.draw_networkx_edges(G,pos,nodelist=[ncenter],alpha=0.4)
    nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),
                           node_size=80,
                           node_color=p.values(),
                           cmap=plt.cm.Reds_r)
    
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    plt.axis('off')
    '''
    #plt.savefig('random_geometric_graph.png')
    plt.show()    