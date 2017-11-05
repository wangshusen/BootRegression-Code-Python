import numpy
from sklearn.datasets import load_svmlight_file

def txt2mat(dataname, isTransformY=False, isRemoveEmpty=False):
    filename = './' + dataname
    X, y = load_svmlight_file(filename)
    X = numpy.array(X.todense())
    n, d = X.shape
    y = y.reshape((n, 1))
    print('Size of X is ' + str(n) + '-by-' + str(d))
    
    mdict = {'A': X, 'b': y}
    
    outfilename = './' + dataname + '.mat'
    scipy.io.savemat(outfilename, mdict)


if __name__ == '__main__':  
    dataname = 'YearPredictionMSD'
    txt2mat(dataname)
    
    
    
