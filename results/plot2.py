import numpy
import matplotlib.pyplot as plt
import scipy.io


#sketch = 'SRHT';
sketch = 'Gaussian';
#sketch = 'Sampling';
dataname = 'ab'
quan = 95;
norm = "l2"
filedir = './'
filename = 'result_' + dataname + '_' + sketch.lower()

startIndex = 3
startLabel = '_t=5d'
ymax = 0.25

def plot(filepath):
    mdict = scipy.io.loadmat(filepath)
    if norm == "infty":
        resultBoot = mdict['resultBootInfty']
        resultEmpirical = mdict['resultEmpiricalInfty']
    else:
        resultBoot = mdict['resultBootL2']
        resultEmpirical = mdict['resultEmpiricalL2']
    tList = mdict['tList']
    tList = numpy.ndarray.flatten(tList)
    
    print(resultEmpirical.shape)
    
    quanTmp = numpy.percentile(resultBoot, quan, axis=2)
    quanBoot = numpy.mean(quanTmp, axis=0)
    quanBootUpper = numpy.percentile(quanTmp, 90, axis=0)
    quanBootLower = numpy.percentile(quanTmp, 10, axis=0)
    quanMul = numpy.percentile(resultEmpirical, quan, axis=0)

    tList = tList[startIndex:]
    quanBoot = quanBoot[startIndex:]
    quanBootUpper = quanBootUpper[startIndex:]
    quanBootLower = quanBootLower[startIndex:]
    quanMul = quanMul[startIndex:]
    
    l = len(tList)
    tTmp = tList / tList[0]
    extra1 = quanBoot[0] / numpy.sqrt(tTmp)
    extra2 = quanBootUpper[0] / numpy.sqrt(tTmp)
    extra3 = quanBootLower[0] / numpy.sqrt(tTmp)
    
    fig = plt.figure(figsize=(3.5, 2.5))

    line0, = plt.plot(tList, quanMul, color='k', linestyle='--', linewidth=4)
    line2, = plt.plot(tList, extra2, color='orange', linestyle='-.', linewidth=2)
    line3, = plt.plot(tList, extra3, color='g', linestyle='-.', linewidth=2)
    line1, = plt.plot(tList, extra1, color='b', linestyle='-', linewidth=2)
    
    plt.scatter(tList[0], extra2[0], s=100, c='orange', alpha=1, marker='*')
    plt.scatter(tList[0], extra3[0], s=100, c='g', alpha=1, marker='*')
    plt.scatter(tList[0], extra1[0], s=100, c='b', alpha=1, marker='*')
    
    fontsize = 10
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Sketch Size t', fontsize=fontsize)
    if norm == "infty":
        plt.ylabel(r"$L_\infty$ Norm Error", fontsize=fontsize)
    else:
        plt.ylabel(r"$L_2$ Norm Error", fontsize=fontsize)
    plt.xticks(fontsize=fontsize) 
    plt.yticks(fontsize=fontsize) 
    #plt.axis([-50, max(tList)+20, 0, ymax])
    plt.title(sketch, fontsize=fontsize+1)
    plt.tight_layout(pad=0.15)

    imagename = filedir + 'extra_' + filename[7:] + '_' + norm + startLabel + '.pdf'
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    #plt.show()


if __name__ == '__main__':  
    filepath = filedir + filename + '.mat'
    plot(filepath)
    
    
    
    