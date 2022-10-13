import MLib as ml

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

    ml.adaBoost(train_data, attribs=attribs, T=10)
    return

if __name__ == "__main__":
    main()