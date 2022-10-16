from copy import deepcopy
import MLib as ml
import numpy as np

def main():
    # initilalization data for bank dataset
    attrib_labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
            'blue-collar','self-employed','retired','technician','services', 'unknown'}, # 'unknown'
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
        'label':{-1, 1} # {'no', 'yes'}
        }
    train_filepath = 'bank/train.csv'
    tests_filepath = 'bank/test.csv'
    numeric_data = {
        'age':['young', 'old'], # this is a numeric value which will be converted to categorical
        'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
        'day':['early', 'late'], # this is a numeric value which will be converted to categorical
        'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
        'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
        'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
        'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
        }
    new_labels = {'no': -1, 'yes': 1}
    
    train_data = ml.importData(train_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)
    tests_data = ml.importData(tests_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)
    del attribs['label'] # remove the label as a valid attribute

    id3 = ml.ID3(train_data, attribs, max_depth=1)
    train_truth = train_data['label'].to_numpy()
    num_train = len(train_truth)
    tests_truth = tests_data['label'].to_numpy()
    num_tests = len(tests_truth)
    aB_train_error = np.zeros(500)
    aB_tests_error = np.zeros(500)
    bT_train_error = np.zeros(500)
    bT_tests_error = np.zeros(500)
    rT_train_error = np.zeros(500)
    rT_tests_error = np.zeros(500)
    for i in np.arange(500):
        aB = ml.adaBoost(train_data, attribs=attribs, T=i+1)
        aB_train_error[i] = sum(train_truth == aB.HFinal(train_data)) / num_train
        aB_tests_error[i] = sum(tests_truth == aB.HFinal(tests_data)) / num_tests

        bT = ml.baggedDecisionTree(train_data, attribs=attribs, T=i+1, m=500)
        bT_train_error[i] = sum(train_truth == bT.HFinal(train_data)) / num_train
        bT_tests_error[i] = sum(tests_truth == bT.HFinal(tests_data)) / num_tests

        rT = ml.randomTree(train_data, attribs=attribs, T=i+1, m=500)
        rT_train_error[i] = sum(train_truth == rT.HFinal(train_data)) / num_train
        rT_tests_error[i] = sum(tests_truth == rT.HFinal(tests_data)) / num_tests

        print(str(i+1) + ", ", end='', flush=True)
    np.savetxt('aB_train_error.csv', aB_train_error, delimiter=',')
    np.savetxt('aB_tests_error.csv', aB_tests_error, delimiter=',')
    np.savetxt('bT_train_error.csv', bT_train_error, delimiter=',')
    np.savetxt('bT_tests_error.csv', bT_tests_error, delimiter=',')
    np.savetxt('rT_train_error.csv', rT_train_error, delimiter=',')
    np.savetxt('rT_tests_error.csv', rT_tests_error, delimiter=',')
    return

if __name__ == "__main__":
    main()