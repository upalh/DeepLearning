"""
This script will take the original adjacency matrix generated by
word2vec_feature_generator.py and explode each of the rows by incorporating
a second hop in the graph.  So basically, what we're doing is for each
original edge (a->b), we find the edges for b, and add them to the list of
edges for a.

This is done by basically performing two map reduce jobs.  The first one
will take the edges for a given vertex (say n) and then share those edges 
with all the other verticies that n form an edge with.  The second job will 
take the edges that it received from the first job and first share its edges
with the sender (so the two basically swap) and if applicable, explode its
edges with the edges it received.  Sometimes this node will not have any
edges in common with the sender (e.g. Amazon co-purchasing network), but
the sender from the first job has an edge with this vertex, so it has to 
know about its edges.  Hence, the first job percolates the information down 
to the edges.  The second job percolates the information up to the senders.  

Once the edges have been swapped, we can then combine the results via the
final reducer.

Here's a quick walkthru of the algorithm.

Input:
z       [a,b,c,d]
a       [e]
b       [f]
c       [g]
d       [h]

mapper1:
z       ([a,b,c,d])
a       (z, [b,c,d])
b       (z, [a,c,d])
d       (z, [a,b,c])

a       ([e])
b       ([f])
c       ([g])
d       ([h])

reducer1:
z       ([a,b,c,d])
a       [([e]),(z, [b,c,d])]
b       [([f]),(z, [a,c,d])]
d       [([h]),(z, [a,b,c])]

a       ([e])
b       ([f])
c       ([g])
d       ([h])

mapper2:
z       ([a,b,c,d])
z       ([e])
z       ([f])
z       ([h])

a       ([e])
b       ([f])
c       ([g])
d       ([h])

reducer2:
z       ([a,b,c,d],[e],[f],[h])
a       ([e])
b       ([f])
c       ([g])
d       ([h])

Created on Fri Feb  7 22:11:59 2014

@author: Upal Hasan

"""

import sys
from mrjob.job import MRJob

class TwoHopFeatureGenerator(MRJob):

    def preprocess_record(self, line):
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
            
    def steps(self):
        """This function defines the steps in our map reduce job. In this
            case, we want to have two job.        
        """
        return [
            self.mr(mapper=self.mapper_percolate_down,
                    reducer=self.reducer_percolate_down),
            self.mr(mapper=self.mapper_percolate_up,
                    reducer=self.reducer_extend_edges)
        ]            
                            
    def reducer_percolate_down(self, key, values):
        """This reducer simply aggregates the information from the first
            mapper before passing them to the second map/reduce job.
            
            Args:
                key: this will be the vetex
                values: the edges for the vertex along with the shared edges.
                    The data structures here are kind of interesting. The
                    original edges will be a list, but all the shared edges
                    will be in a dictionary where the key is the sender
                    vertex and the values are all of the sender's edges.
                    
            Returns:
                key: corrresponds to the vertex name
                values: aggregated values as a list. This list will contain
                    one list for the original edges, and potentially several
                    dictionaries each corresponding to a sender that has an
                    edge with this vertex.
        """
        toSendList = [element for element in values]
        #print "1st reducer: " + key + " values: " + str(toSendList) + "\n"
        yield key, toSendList
                            
    def mapper_percolate_down(self, _, line):
        """This mapper will percolate its edges down to everyone that it
            shares an edge with.  Additionally, it will emit its own edges, 
            so that it does not get lost later on in the map/reduce jobs.
            
            Args:
                _: None
                line: a line representing the adjancecy matrix for a given
                    node.
                    
            Returns:
                There are two yields here. For the first:
                    element: corrresponds to the vertex name to send the shared
                        edges to.
                    toWriteList: a dictionary describing the current node's 
                        verticies.
                For the second:
                    key: corresponds to the current vertex's name
                    parsedLine: the edges for the current vertex.                    
        """
        
        vertex, edges = self.preprocess_record(line)
        if not edges:
            raise Exception("parsed line is empty. this should not happen")

        # traverse through the edges and emit the list to everyone that 
        # we form an edge with. they will use these edges to explode their
        # edge list and will also share their edges with us.         
        for edge in edges:
            edgeDictionary = {}            
            #print "parsedLine: " + str(edges) + "\n"
            listToSend = [el for el in edges if el != edge]
            #print "listToSend: " + str(listToSend) + "\n"
            if listToSend:
                edgeDictionary[vertex] = listToSend
                #print "element: " + edge + " vals: " + str(listToSend) + "\n"
                yield edge, edgeDictionary

        yield vertex, edges
        #print "sent main record: key " + vertex + " values: " + str(edges) + "\n"


    def mapper_percolate_up(self, vertex, edgeList):
        """This mapper will percolate a vertex's edges up to everyone that 
            shared their edges with them in the first map/reduce job. This
            will not automatically happen in the first job if this vertex
            does not share an edge with the sender (in the case of the
            Amazon co-purchasing network dataset).
            
            Howeve, if it shares an edge with the sender, it will explode
            its list of edges and emit that out.
            
            Args:
                vertex: the name of the vertex
                edgeList: a list containing the edges of the vertex and also
                    a set of dictionaries describing the edges of the verticies
                    that shares an edge with this vertex.
                    
            Returns:
                There are two yields here. For the first:
                    element: corrresponds to the vertex name to send the shared
                        edges to.
                    toWriteList: a dictionary describing the current node's 
                        verticies.
                For the second:
                    key: corresponds to the current vertex's name
                    parsedLine: the edges for the current vertex.                    
        """

        keyList = [] 
        valueList = []        
        toSendList = []
        #print "tupleKey: " + vertex + " tupleList: " + str(edgeList) + "\n"
        
        # here we just want to emit out a vertex's set of edges if there's
        # nothing to explode (i.e. it doesn't share an edge with anyone else)
        if len(edgeList) == 1 and type(edgeList[0]) == list:
            yield vertex, edgeList[0]
        else:
            # in this case we want to separate out the set of keys to send
            # edges to, the set of edges that we may need to use to 
            # explode our own edge list, and our own edges that we will need
            # to send
            for keyTuple in edgeList:
                if type(keyTuple) == list:
                    toSendList = keyTuple
                elif type(keyTuple) == dict:
                    for k, v in keyTuple.iteritems():
                        keyList.append(k)
                        valueList.append(v)
                else:
                    raise Exception("type in dictionary not supported: " + \
                        str(type(keyTuple)))
    
            if toSendList:                
                # if there are any edges to send, go ahead and send to
                # each of the verticies that shared with us.                
                for key in keyList:
                    yield key, toSendList
                
                # and now determine our new expanded list (if applicable)
                # based on what the other verticies shared with us. we have
                # to see if any of the keys (names of veticies that shared
                # with us) is present in our edge list. if it is, then we
                # know there was a shared edge, so we can add the shared 
                # edges to our list.
                expandedList = []
                expandedList.extend(toSendList)
                for i, keyTuple in enumerate(keyList):
                    if keyTuple in expandedList:
                        expandedList.extend(valueList[i])
                    
                yield vertex, expandedList        

            
    def reducer_extend_edges(self, vertex, edgeList):
        """This reducer will combine all the list of edges together. By 
            this point everything in values should be combined, so there's
            no complicated logic here.
            
            Args:
                vertex: the name of the vertex
                edgeList: a list containing the edges of the vertex and all
                    the new edges that will need to be added to it.
                    
            Returns:
                vertex: the name of the vertex.
                expandedEdges: a list of new edges after the explosion.
        """
        
        expandedEdges = []
        for edge in edgeList:
            expandedEdges.extend(edge)
        
        yield vertex, str(list(expandedEdges))
        
if __name__ == '__main__':
    TwoHopFeatureGenerator.run()
