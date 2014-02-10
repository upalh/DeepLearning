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
    for node in G.adj.keys():
    	adjacent = G.adj[node].keys()
    	q = Queue.Queue()
    	seen = {}
    	validNeighbors = []
    	for adj in adjacent:
    		q.put(adj)
    		seen[adj] = True

    	MAX_SIMILARS_PER_DOC = 25
    	while not q.empty():
    		curr = q.get()
    		# add all the similars that have a valid node in the graph:
    		if curr in G.node:
    			validNeighbors.append(curr)

    			#If we haven't reached threshold of neighbors add any neighbors into the graph
    			if(len(validNeighbors) < MAX_SIMILARS_PER_DOC):
	    			nextNeighbors = G.adj[curr].keys()
	    			for adj in nextNeighbors:
	    				if adj not in seen:
	    					q.put(adj)
	    					seen[adj] = True
		validNeighbors.append(node)
    	of.write(json.dumps({"asin": node, "similar": validNeighbors}))
    of.close()