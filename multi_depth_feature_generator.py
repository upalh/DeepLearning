import sys
import json
import networkx as nx
import Queue
   
if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        path_to_file = sys.argv[1]
    else:
        path_to_file = "data/amazon_raw.json"

    if(len(sys.argv)>2):
        path_to_output_file = sys.argv[2]
    else:
        path_to_output_file = "data/deep_features.json"

    G = nx.Graph()
    
    count = 0
    
    # Build graph from file:
    i = 0
    for record in open(path_to_file):    
        record = json.loads(record)
        if not "similar" in record or not "title" in record:
        	continue
        
        asin = record["ASIN"]
        title = record["title"]
        similar = record["similar"]
    	
        G.add_node(asin, {"title" : title})
        for adjASIN in similar:
        	# check if adjacent node exists:
        	G.add_edge(asin, adjASIN)
        if i % 10000 == 0:
        	print "Loaded " + str(i) + " nodes"
        i = i + 1



    # Print basic statistics on graph:
    print "Computing largest sub component"
    mainGraph = nx.connected_component_subgraphs(G)[0]
    print "Largest connected component:"
    print nx.number_of_nodes(mainGraph)

    # Now try to generate feature vectors by doing "depth charges" 
    of = open(path_to_output_file, "w")    
    i = 0
    for node in mainGraph.adj.keys():
        adjacent = mainGraph.adj[node].keys()
        q = Queue.Queue()
        seen = {}
        validNeighbors = [node]
        seen[node] = True
        
        for adj in adjacent:
            q.put(adj)
            seen[adj] = True
            
        MAX_SIMILARS_PER_DOC = 10000
        while not q.empty() and len(validNeighbors) < MAX_SIMILARS_PER_DOC:
            curr = q.get()
            # add all the similars that have a valid node in the graph:
            if curr in mainGraph.node:
                validNeighbors.append(curr)
                #If we haven't reached threshold of neighbors add any neighbors into the graph
                nextNeighbors = mainGraph.adj[curr].keys()
                for adj in nextNeighbors:
                    if adj not in seen:
                        q.put(adj)
                        seen[adj] = True	
        if i % 10000 == 0:
            print "Generated " + str(i) + " feature vectors"
        of.write(json.dumps({"asin": node, "similar": validNeighbors})+"\n")
        i = i + 1
    
    of.close()