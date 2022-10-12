import MLib as ml
import numpy as np
import pandas as pd

def main():
    # initilalization data for bank dataset
    attrib_labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
            'blue-collar','self-employed','retired','technician','services'}, # 'unknown'
        'marital':{'married','divorced','single'},
        'education':{'secondary','primary','tertiary', 'unknown'},  # 'unknown'
        'default':{'yes', 'no'}, 
        'balance':{'low', 'high'}, # this is a numeric value which will be converted to categorical
        'housing':{'yes', 'no'}, 
        'loan':{'yes', 'no'}, 
        'contact':{'telephone', 'cellular', 'unknown'}, #'unknown'
        'day':{'early', 'late'}, # this is a numeric value which will be converted to categorical
        'month':{'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}, 
        'duration':{'short', 'long'}, # this is a numeric value which will be converted to categorical
        'campaign':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'pdays':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'previous':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'poutcome':{'other', 'failure', 'success', 'unknown'}, # 'unknown'
        'label':{'no', 'yes'}
        }
    data_labels = {'yes', 'no'}
    train_filepath = 'bank/train.csv'
    test_filepath = 'bank/test.csv'
    numeric_data = {
        'age':['young', 'old'], # this is a numeric value which will be converted to categorical
        'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
        'day':['early', 'late'], # this is a numeric value which will be converted to categorical
        'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
        'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
        'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
        'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
        }
    index_col = 16 
    max_depth = 1
    
    training_data = ml.importData(train_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)
    test_data = ml.importData(test_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)

    tree = ml.ID3(training_data, attribs, None, 'entropy', max_depth) # build the tree
    ml.printTree(tree) # crude print of tree
    print('Avg Error Training Dataset = ' + str(ml.treeError(tree, training_data))) # test the error for the given dataset
    print('Avg Error Test Dataset = ' + str(ml.treeError(tree, test_data))) # test the error for the given dataset

    return

if __name__ == "__main__":
    main()