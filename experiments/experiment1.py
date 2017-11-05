import numpy
import scipy.io
import scipy.sparse.linalg

def loaddata(path):
    dict = scipy.io.loadmat(path)
    return dict['A'], dict['b']

def sketching(a, b, t, sketch):
    n = a.shape[0]
    if sketch == 'gaussian':
        s = numpy.random.randn(t, n)
        aSketch = numpy.dot(s, a) / numpy.sqrt(t)
        bSketch = numpy.dot(s, b) / numpy.sqrt(t)
        
    return aSketch, bSketch


def experiment(a, b, tList, sketch, numRepeat, numBoot):
    wOpt = scipy.sparse.linalg.lsmr(a, b, atol=1E-8, btol=1E-8)[0]
    tLen = len(tList)
    tMax = numpy.max(tList)
    resultEmpiricalL2 = numpy.zeros((numRepeat, tLen))
    resultEmpiricalInfty = numpy.zeros((numRepeat, tLen))
    numIterEmpirical = numpy.zeros((numRepeat, tLen))
    resultBootL2 = numpy.zeros((numRepeat, tLen, numBoot))
    resultBootInfty = numpy.zeros((numRepeat, tLen, numBoot))
    numIterBoot = numpy.zeros((numRepeat, tLen, numBoot))
    
    for r in range(numRepeat):
        aSketch, bSketch  = sketching(a, b, tMax, sketch)
        
        for i in range(tLen):
            t = tList[i]
            aSketchT = aSketch[0:t, :]
            bSketchT = bSketch[0:t, :]
            results = scipy.sparse.linalg.lsmr(aSketchT, bSketchT, atol=1e-6, btol=1e-6, maxiter=1E4)
            wT = results[0]
            
            # matrix multiplication errors
            res = wT - wOpt
            resultEmpiricalL2[r, i] = numpy.linalg.norm(res)
            resultEmpiricalInfty[r, i] = numpy.max(numpy.abs(res))
            numIterEmpirical[r, i] = results[2]
            
            # bootstrap errors
            for boot in range(numBoot):
                idx = numpy.random.choice(t, size=t, replace=True)
                aTilde = aSketchT[idx, :]
                bTilde = bSketchT[idx, :]
                results = scipy.sparse.linalg.lsmr(aTilde, bTilde, atol=1e-6, btol=1e-6, maxiter=1E4, x0=wT)
                wTilde = results[0]
                res = wT - wTilde
                resultBootL2[r, i, boot] = numpy.linalg.norm(res)
                resultBootInfty[r, i, boot] = numpy.max(numpy.abs(res))
                numIterBoot[r, i, boot] = results[2]

        avgIterEmpirical = numpy.mean(numIterEmpirical[r, :])
        avgIterBoot = numpy.mean(numpy.mean(numIterBoot[r, :, :]))
        print('Iteration ' + str(r) + ': average iter of solvers are ' + str(avgIterEmpirical) + ' and ' + str(avgIterBoot))

    mdict = {'tList': tList,
             'resultEmpiricalL2': resultEmpiricalL2,
             'resultEmpiricalInfty': resultEmpiricalInfty,
             'numIterEmpirical': numIterEmpirical,
             'resultBootL2': resultBootL2,
             'resultBootInfty': resultBootInfty,
             'numIterBoot': numIterBoot}
    return mdict

if __name__ == '__main__':
    #dataname = 'YearPredictionMSD'
    dataname = 'abalone'
    inputpath = '../data/' + dataname + '.mat'
    sketch = 'gaussian'
    outputpath = '../results/result_' + dataname[0:2] + '_' + sketch + '.mat'
    numBoot = 20
    numRepeat = 100
    
    a, b = loaddata(inputpath)
    n, d = a.shape
    print('Data loaded! n=' + str(n) + ', d = ' + str(d))
    tList = numpy.arange(2*d, 30*d, d)
    
    mdict = experiment(a, b, tList, sketch, numRepeat, numBoot)
    scipy.io.savemat(outputpath, mdict)
    