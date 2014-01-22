"""
File to do some simple plotting on the number of purchase distributions
over the products.  The data that I used for the plots were not checked in
because they're too big for the repo.

From the plot, one can see that the distribution decreases exponentially
as the number of purchases for a given product increases.  The structure
of this plot is very similar to the distributions of number of users that
have purchased something.  So most users have bought few products, but few
have bought a lot.

Created on Mon Jan 20 07:50:52 2014

@author: ubuntu

"""

import sys
import pylab as pl

# this will dictate how many bars we've got. based on the data,
# the number of purchases for most products ranges from 0 - 99, so
# i just have a 100 buckets. each bucket will store a count of the number
# of products that have been bought X times, where X varies from 0 - 99.
# for anything where the number of products that have been bought exceeds
# 99, i dump them all into the last bucket.
NUM_BUCKETS = 100

if __name__ == "__main__":
    
    if(len(sys.argv)>1):
        path_to_file = sys.argv[1]
    else:
        path_to_file = "numPurchaseDistributions"

    # create our label vectors
    labels = []
    for label in range(0, NUM_BUCKETS):
        labels.append(label)
        
    # now calculate the number of products for each of our buckets.
    bucketCounts = dict()
    for line in open(path_to_file):
        x, y = line.split()

        def insert(bucket):
            if str(bucket) not in bucketCounts:
                bucketCounts[str(bucket)] = 1 
            else:
                bucketCounts[str(bucket)] = bucketCounts[str(bucket)] + 1
            
        if int(y) > NUM_BUCKETS:
            insert(NUM_BUCKETS)
        else:            
            bucket = int(y) % NUM_BUCKETS
            insert(bucket)

    # get the counts in a proper order in the list to plot
    values = []
    for label in range(0, NUM_BUCKETS):
        value = bucketCounts[str(label)] if str(label) in bucketCounts else 0
        values.append(value)
        
    fig = pl.figure()
    ax = pl.subplot(111)
    ax.bar(labels, values)
    
    #Create a y label
    ax.set_ylabel('Number of products')
     
    # Create a title, in italics
    ax.set_title('Number of purchases',fontstyle='italic')
     
    # This sets the ticks on the x axis to be exactly where we put
    # the center of the bars.
    ax.set_xticks(range(NUM_BUCKETS))
    
    pl.show()