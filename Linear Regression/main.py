from operator import index
import pandas as pd
import numpy as np

def LMSRegression(S, r=.001, threshold=.001, label_name=''):
    yi = S.pop(label_name).to_numpy() # seperate off output column
    m = len(S) # number of examples in X
    d = len(S.columns) + 1 # number of dimensions to X per example
    w = np.zeros([1,d]) # init weights to zero (+1 because first slot in x is a 1)
    X = np.concatenate([np.ones([1,m]).T, S.to_numpy()],axis=1)
    error = np.inf
    while (error > threshold):
        for j in np.arange(d): # for all dimensions in X
            w = w - r * calc_dj_dw(yi, w, X, X[:,j])
    return
def calc_dj_dw(yi, wT, xxi, xj):
    m = len(xj)
    temp = 0
    for i in np.arange(m):
        temp += sum(( yi - np.sum(np.multiply(wT, xxi),axis=1)) * xj[i])
    return -temp

def main():
    data_filepath = "./concrete/slump_test.csv"
    terms = pd.read_csv(data_filepath, sep=',', header=0, index_col=0) # read in the csv file into a DataFrame object , index_col=index_col
    del terms['FLOW(cm)'] # unused outputs
    del terms['Compressive Strength (28-day)(Mpa)'] # unused outputs
    LMSRegression(terms, r=1, threshold=.001, label_name='SLUMP(cm)')
    return

if __name__ == "__main__":
    main()