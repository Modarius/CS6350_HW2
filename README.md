This is a machine learning library developed by Alan Felt for CS5350/6350 at the University of Utah
To learn a decision tree you need to setup a few things.
attrib_labels is a list of labels of the attributes in the data set
attrib_labels = ['age', 'job', 'marital'...]

attribs is a dictionary with sub-lists of the attributes in each label
attribs = {
    'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
    'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
        'blue-collar','self-employed','retired','technician','services', 'unknown'}, # 'unknown'
    'marital':{'married','divorced','single'},
    'label':{-1, 1} # {'no', 'yes'}
    }

filepaths for the data 
train_filepath = 'bank/train.csv'
tests_filepath = 'bank/test.csv'

if the data has an numeric values you can enter the label here and 2 attributes to split the numeric data on
numeric_data = {
    'age':['young', 'old'], # this is a numeric value which will be converted to categorical
    'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
    'day':['early', 'late'], # this is a numeric value which will be converted to categorical
    'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
    'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
    'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
    'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
    }
    
if the data is labeled with output labels you'd like to change, put them here. the key is the old label, the value is the new label
new_labels = {'no': -1, 'yes': 1}

then you can import your data like so:
train_data = ml.importData(train_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)
tests_data = ml.importData(tests_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)

Following import, delete the label as a valid attribute, it is an output not an attribute
del attribs['label'] # remove the label as a valid attribute

You are then ready to call the main ID3 algorithm or one of the algorithms:
tree = ml.ID3(training_data, attribs, None, 'entropy', max_depth) # build the tree
