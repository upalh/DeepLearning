'''
This file basically will take the amazon co-purchasing graph and construct
a pre-processed version of the data.  This data will then be read and be
used to construct a list of sentences for word2vec to process.

Although this parser was created for the amazon co-purchasing graph, it 
should work for most datasets where the records are multi-lines and are
stored as <key><delimiter><value> with some other delimiter that will 
separate out the different records.
 
To customize this for other datasets, we can construct a schema for the
data and if the data is ordered properly, this file will take the original
dataset and construct a JSON version of the fields of interest,
where each line represents a record.

The resulting file can then be used by hadoop to further pre-process and
pass to word2vec.  The goal of this script is to take a multi-line, key-value,
paired dataset, and convert it into a standard format that our other scripts
will be able to work with.

Here's a sample schema below.  For each field, a user must specify
the type of the field, and the "keep" attribute.  The "keep" attribute
specifies the fields of interest.
    
    {
        "datasetPath": "amazon-meta-test.txt",
        "multiline": "true",
        "betweenRecordDelimiter": "\r\n",
        "skipLines": 3,
        "betweenKeyValDelimiter": ":",
        "fields": {
        	   "Id": {
                 "type": "numeric"
        	    },
               "ASIN": {
        		  "type": "string",
        		  "keep": "true"
        	    },
        	    "title": {
        		  "type": "string",
        		  "keep": "true"
        	    },
        	    "group": {
        		   "type": "string"
        	    },
        	    "salesrank": {
                 "type": "numeric"
        	    },
              "similar": {
        		  "type": "list",
        		  "hasLength": "true",
        		  "element_type": "numeric",
        		  "firstDelimiter": "\\s+",
        		  "keep": "true"
        	    },
             "reviews": {
        		  "hasLength": "true",
        		  "type": "list-multiline",
        		  "element_type": "string",
        		  "firstDelimiter": ":\\s+",
        		  "keep":"true",
        		  "keepIndicies": "1"
        	    }    
        }
    }

For fields which are of type "list" or "list-multiline" a user must
specify the "firstDelimiter" attribute.  This specifies how the elements
are separated.  In an inline list, where all the elements are on one line,
we have to tell the parser how they are separated, so we can split them up.
The firstDelimiter can be a regular expression to help with that.  

The user can also specify which indicies to keep of that list by specifying
the "keepIndicies" attribute.  The indicies must be comma spearated.

To make this work for other datasets, the user must override the following 
methods:
        get_list_length - this function is used by the program to generate
            a list from the elements that are listed multiline. Basically 
            when trying to process a list of list that is specified over
            multiple lines, the program needs to know how many inner lists
            there will be. This function provides that information.
            
            For inline lists, the program will assume that the first element
            is length of the list IF the field has the "hasLength" attribute 
            is set to "true."  The program can work with the data if the 
            length of the list is specified before the data itself 
            (e.g. 6 0 1 2 3 4 5, where the 6 in the beginning is the length 
            of the list). If the length is not specified, the program will 
            store all the data, otherwise it will take everything AfTER the
            first element.
            
        post_process_element - this function can be tweaked for a given
            dataset if we want further control for how to parse elements of
            a list.
            
TODO:
    Add the "secondDelimiter" attribute to do further preprocessing of the
    data after the elements of the list have been split.
    
    Make more general without having to tweak.
    
    The program will still hold the condensed representation in memory 
    until it's time to write it out. Need to explore if there's a way to
    write it out as we go to reduce the memory footprint.    
    
    The only assumption of this script is that the dataset is key-value
    pairs and is multiline.  It can also be tweaked slightly to work with 
    single lined key-value paired data.
    
Created on Jan 17, 2014

@author: mhasan
'''

import sys
import re
import os
import json


# stores the base directory of data that we will be processing and gets
# initialized by the command line.
DATA_DIR_PATH = ""

NOT_MULTILINE = "false"

def get_schema_value_for_key(schema, key):
    """Retrieves the contents of the schema for a given key.
    
    Args:
        schema: our schema representing the dataset being processed
        key: the section of the schema to retrieve
    
    Returns:
        We should be getting back a dictionary mapping of the sub-fields
        of the schema corresponding to the sections as given by key.
        
        If the key doesn't exist in the schema, this function will return 
        None    
    """
    if key in schema:
        return schema[key]
 
def get_fields_schema(schema):
    """Retrieves the contents of the schema for the "fields" key.
    
    Args:
        schema: our schema representing the dataset being processed
    
    Returns:
        We should be getting back a dictionary mapping of the sub-fields
        of the schema corresponding to the sections of the "fields" key. 
        This part of the schema provides details about each of the fields
        of our dataset.
        
        If the key doesn't exist in the schema, this function will return 
        None    
    """
    
    return get_schema_value_for_key(schema, "fields")
    
def get_keep_fields(schema):
    """Retrieves the contents of the schema for the data fields that
    we are interested in running an analysis on. This is specified by
    the "keep" attribute for each field.
    
    Args:
        schema: our schema representing the dataset being processed
    
    Returns:
        We should be getting back a dictionary mapping of the sub-fields
        of the schema corresponding to the fields of the dataset that has
        the "keep" attribute set to "true". 
        
        If the key doesn't exist in the schema, this function will return 
        None    
    """    
    keepFields = []
    fieldsSchema = get_fields_schema(schema)
    if not fieldsSchema:
        raise Exception("schema must have fields attribute")
        
    for fieldName, fieldValue in fieldsSchema.iteritems():
        if "keep" in fieldValue and fieldValue["keep"] == "true":
            keepFields.append(fieldName)
    return keepFields

def read_line(fileHandle):
    """Retrieves a line of the file. This method exists incase we need to 
    do some kind of pre-processing to get a new line from the open file.
    Originally I wanted to use this function to obtain a line of the file
    using the "yield" keyword to ensure that the entire file doesn't get 
    loaded into memory, but then later found out that the open() already
    lazily loads the lines, so there was nothing to worry about. However,
    I decided to leave this in incase we need to use it later.
    
    Args:
        fileHandle: the handle to the file that is open.
    
    Returns:
        A string representing the next line of the file.
    """    
    
    return fileHandle.readline()
    
def read_record(fileHandle, readLineCallback, schema, multiline, delim):
    """Reads in one record of the file that we're processing. The assumption
    is that the records are multiline and are separated by a delimiter.
    
    Args:
        fileHandle: the handle to the file that is open.
        readLineCallback: the callback function to process the line that's
            been read in.
        schema: the schema representing the dataset that we're processing.
        multiline: indicates if we're processing a multiline record or 
            single line record.
        delim: the delimiter that seperates the different records. This
            value is defined in the schema object.
    Returns:
        A dictionary mapping of the data fields of interest as specifed by 
        the "keep" attribute. If we are interested in keeping a "list" 
        field, the key will map to a list object.
    """    
    
    record = dict()
    while True:
        try:
                
             line = read_line(fileHandle)
             # check for end of file
             if not line:
                 break

             # need to check for the end of a record
             if is_done(line, delim) or multiline == NOT_MULTILINE:
                 return [True, record]             
                 
             # the callback will define how we want to process each line as it's
             # read in.         
             lineSplit = readLineCallback(line, schema)
             
             # couldn't parse the line because the field didn't exist in the
             # schema. could be mistyping somewhere in the dataset, so skip it.
             if not lineSplit:
                 continue
             
             fieldSchema = get_schema_value_for_key(get_fields_schema(schema), lineSplit[0])
             valueType = get_schema_value_for_key(fieldSchema, "type")
             valueDelimiter = get_schema_value_for_key(fieldSchema, "firstDelimiter")
             keepField = get_schema_value_for_key(fieldSchema, "keep")
             hasLength = get_schema_value_for_key(fieldSchema, "hasLength")
             hasLength = hasLength if hasLength else "false"
             
             # if we don't want to keep the field for further processing, move
             # onto the next line in the record.
             if not keepField:
                 continue
             
             # make sure to convert the unicode to strings
             lineSplit[1] = lineSplit[1].encode("utf8")
             
             # now process the field accordingly. if it's a list, we have to
             # do specialized processing, otherwise, just store it in the 
             # dictionary
             if valueType == "list":
                 record[lineSplit[0]] = handle_single_line_list(lineSplit[1], 
                    valueDelimiter, fieldSchema, hasLength)
             elif valueType == "list-multiline":
                 record[lineSplit[0]] = handle_multi_line_list(lineSplit[1], 
                    valueDelimiter, fieldSchema, fileHandle, hasLength)             
             else:
                 record[lineSplit[0]] = lineSplit[1]                  
        except Exception:
            print "error parsing: %s" % line

    # this shouldn't happen
    return [False, record]

def extract_key_value(lineSplit, schema):
    """This method will gather up the values of a list based on the fields of
        as specified by the "keepIndicies" attribute. If this attribute is
        not specified, the program will take all the indicies.
    
    Args:
        lineSplit: the elements of the index after they have been split by
            the delimiter as specified by the "firstdelimiter."
        schema: the schema representing the dataset that we're processing.
        
    Returns:
        A list of field values corresponding to the indicies as specified by
        the "keepIndicies" attribute.
        
        TODO: secondary delimeter if user wants to further refine the key-val
            pairs.        
    """    
    
    indiciesToRetain = get_schema_value_for_key(schema, "keepIndicies")
    indiciesSplit = indiciesToRetain.split(",") if indiciesToRetain else None
    
    values = []
    if indiciesSplit is None:
        for idx,val in enumerate(lineSplit):
            values.append(post_process_element(val))
    else:
        for idx,val in enumerate(lineSplit):
            if str(idx) in indiciesSplit:
                values.append(post_process_element(val))
                
    return values
        
def handle_multi_line_list(value, valueDelimiter, fieldSchema, fileHandle, hasLength):    
    """This method will properly parse a multi-line list of lists, where
        we define the number of lists on the first line and then list
        each individual list afterwards.
    
    Args:
        value: the string version of the list. that is the entire list is
            specified as a string that is seperated by the valueDelimiter.
        valueDelimiter: the elements of the index after they have been split by
            the delimiter as specified by the "firstdelimiter" attribute for
            the field currently being processed.
        fieldSchema: the schema representing the dataset correspoding to 
            the "fields" of the datasets.
        fileHandle: the handle to the file.close
        hasLength: specifies if the number of lists is specified in the data.
            if it is not, then the program will raise an exception because
            it's hard to parse.
        
    Returns:
        A list of lists that corresponds to the set of parsed lists.
        
    """    
    
    if not hasLength:
        print "need to provide length for multi-line list object"
        raise Exception()
    
    i = 0 
    arrayElements = []       
    length = int(get_list_length(value))
        
    while i < length:
        element = read_line(fileHandle)
        # each inner list of the multi-line list does not have the
        # hasLength attribute set
        arrayElements.append(handle_single_line_list(element, 
                valueDelimiter, fieldSchema, "false"))
        
        i = i+1
                        
    return arrayElements
   
def handle_single_line_list(value, valueDelimiter, fieldSchema, hasLength):
    """This method will properly parse a single-line list.
    
    Args:
        value: the string version of the list. that is the entire list is
            specified as a string that is seperated by the valueDelimiter.
        valueDelimiter: the elements of the index after they have been split by
            the delimiter as specified by the "firstdelimiter" attribute for
            the field currently being processed.
        fieldSchema: the schema representing the dataset correspoding to 
            the "fields" of the datasets.
        hasLength: specifies if the number of elements of the list is 
            specified in the data.
            if it is not, then the program will keep all the elements. if
            it is, then it assumes that the length is specified as the first
            element, and will take all the elements after the first one.
        
    Returns:
        A list that corresponds to the elements of the single line lists.        
    """        
    preprocess = lambda x: x.strip()
    lineSplit = map(preprocess, [y for y in re.split(valueDelimiter, value)])
    
    # now we need to extract out the relevant information
    if hasLength == "true":
        return extract_key_value(lineSplit[1:], fieldSchema)
        
    return extract_key_value(lineSplit, fieldSchema)
    
def load_records(path, schema, processLine, modulo = 1000):
    """This method will load all the fields of inerest from the dataset.
    This is esentially the main loop of the program.
    
    Args:
        path: the path to the file that we want to load up. if the file 
            can't be opened, the program will throw an exception.
        schema: the schema representing the dataset.
        processLine: the callback function that will be called for each line
            that is read in.
        modulo: parameter to help with debugging. prints out a record every
            modulo times.
        
    Returns:
        A list of dictionaries that corresponds to all the data elements. 
        Each dictionary corresponds to a single record.        
    """        
    
    records = []

    # figure out if there are any fields that we want to skip first
    skipLines = get_schema_value_for_key(schema, "skipLines")
    skipLines = int(skipLines) if skipLines else 0      
        
    with open(path) as fileHandle: 
        if not fileHandle:
            print "could not open file"
            raise Exception()    
            
        lineNumber = 0
        while lineNumber < skipLines:
            fileHandle.readline()
            lineNumber = lineNumber + 1            
            
        multiline = get_schema_value_for_key(schema, "multiline")    
        multiline = multiline if multiline else "false"     
        
        #ignored if we don't have a multi-line record dataset
        delim = get_schema_value_for_key(schema, "betweenRecordDelimiter")    
        delim = delim if delim else None    

        # loop to keep reading records and keeping a list of the parsed data.                              
        count = 0
        while True:
            moreRecords, record = read_record(fileHandle, processLine, \
                schema, multiline, delim)
                
            if not moreRecords:
                break
            records.append(record)
            
            if count % modulo == 1:
                print "processed record #: " + str(count)
            count = count + 1

    return records
        
def save_records(path, records, schema, modulo=1000):
    """This method will save all the parsed data to disk. It will save the
        data in a comma separated list, where each element will correspond
        to a field that has the "keep" attribute set to "true." If the 
        "keep" attribute of a field is set to "true" but the data does
        not exist for it, then we substitute "NA" for it.
    
    Args:
        path: the path to the file that we want to save to.
        records: this is the list of dictionary mappings that correspond to
            all the parsed data.
        schema: the schema representing the dataset.
        modulo: parameter to help with debugging. prints out a record every
            modulo times.
        
    Returns:
        This function doesn't return anything, but will save a file to disk
        as specified by the path parameter.        
    """            
        
    count = 0    
    with open(path, "w") as fileHandle:
        for record in records:
            json.dump(record, fileHandle)
            fileHandle.write("\n")
            if count % modulo == 1:
                print "wrote record: %s" % str(count)
            count = count + 1        
            
def process_line(line, schema): 
    """This is the callback method for every line that will read in. It will
        perform the first split to seperate out key from value and then 
        return the split data as <key, value>. To perform the first split,
        the program looks at the value of the "betweenKeyValDelimiter" 
        value.  If this is not set, we assume that there's a " " to 
        seperate them out. We also assume that the first element is the key
        and the latter values are the value.  if after the first split, we
        get a list for the value, then we combine that list into a string
        and pass that back.
    
    Args:
        line: one line as string.
        schema: the schema representing the dataset.
        
    Returns:
        A list of length 2 represented by <key, value>.
    """            
    
    keyValSeperatorChar = get_schema_value_for_key(schema, "betweenKeyValDelimiter")
    keyValSeperatorChar = keyValSeperatorChar if keyValSeperatorChar else " "     
        
    preprocess = lambda x: x.strip()
    lineSplit = map(preprocess, line.split(keyValSeperatorChar))

    fieldKey = lineSplit[0] if lineSplit else ""
    fieldSchema = get_schema_value_for_key(get_fields_schema(schema), fieldKey)    
    if not fieldSchema:
        #print "field name %s does not exist." % (fieldKey)
        return None
        #raise Exception(errMessage)
        
    return [fieldKey, lineSplit[1] if type(lineSplit[1]) == str else ' '.join(lineSplit[1:])]

########### functions to override 
def is_done(line, delim):
    """This function checks for the end of a record.  This is a function that
        can be overriden to customize the end of a record.
    
    Args:
        line: one line as string.
        schema: the delimiter specifying the end of a record..
        
    Returns:
        A boolean specifying if we're at the end of the record or not.
    """            
    
    return line == delim

def get_list_length(value):
    """This function enables the user to customize how to parse the values
        of the initial multiline list entry to get the number of inner
        lists. This was too hard to generalize, so it might be best to just
        provide the hook and user the user to tailor it to their dataset.
    
    Args:
        value: the value portion a line after the initial split.
        
    Returns:
        An int specifying the number of lists to parse.
    """            
    
    # this is the hacky part that will need to be overriden some how for 
    # new datasets since the format can vary so wildly for each dataset
    preprocess = lambda x: x.strip()
    lineSplit = map(preprocess, [y for y in re.split("\s+", value)])

    if lineSplit:
        return int(lineSplit[3])
        
def post_process_element(value):
    """This function enables the user to customize how to parse a value
        of the list entries. This was too hard to generalize, so it might 
        be best to just provide the hook and user the user to tailor it 
        to their dataset.
    
    Args:
        value: the value portion a line after the initial split.
        
    Returns:
        An string representation of the field after customized parsing.
    """            
    
    # this is another method that will need to be overriden some how for 
    # new datasets since the format can vary so wildly for each dataset
    return re.split("\s+", value)[0]                    
########### functions to override 
 
def get_data_path(schema):
    """This function returns the path to the data. if the data file
        does not exist, the program will raise an exception.
    
    Args:
        schema: the schema representing the dataset.
        
    Returns:
        A path for the dataset to load.
    """            
    
    dataSetPath = get_schema_value_for_key(schema, "datasetPath")
    if not dataSetPath:
        errMessage =  "must provide dataset path in json file"
        raise Exception(errMessage)
    
    return os.path.join(DATA_DIR_PATH, dataSetPath)

def get_save_path(schema):
    """This function returns the path for saving the data. 
    
    Args:
        schema: the schema representing the dataset.
        
    Returns:
        A path for save the preprocessed data.
    """            
    
    dataSetPath = get_schema_value_for_key(schema, "datasetPath")
    if not dataSetPath:
        errMessage =  "must provide dataset path in json file"
        raise Exception(errMessage)
    
    return os.path.join(DATA_DIR_PATH, os.path.dirname(dataSetPath), 
            os.path.basename(dataSetPath.split(".")[0] + "_preprocessed.out"))

def debug(records, fileName = "debug.out"):
    
    with open(fileName, "w") as fileHandle:
        for record in records:
            if "reviews" in record and not record["reviews"]:
                fileHandle.write(record["Id"] + "\n")

if __name__ == '__main__':
    
    if(len(sys.argv)>1):
        path = sys.argv[1]
    else:
        path = "/home/ubuntu/Desktop/data/amazon/schema.txt"
        
    DATA_DIR_PATH = os.path.dirname(path)
    schema = json.load(open(path))
    
    records = load_records(get_data_path(schema), schema, process_line)
    save_records(get_save_path(schema), records, schema)
    
    #debug(records)
    #print str(schema)