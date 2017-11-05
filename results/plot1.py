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

startIndex = 2
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
    quanMul = numpy.percentile(resultEmpirical, quan, axis=0)

    tList = tList[startIndex:]
    quanBoot = quanBoot[startIndex:]
    quanMul = quanMul[startIndex:]

    fig = plt.figure(figsize=(3.5, 2.5))

    line0, = plt.plot(tList, quanMul, color='k', linestyle='--', linewidth=3)
    line1, = plt.plot(tList, quanBoot, color='r', linestyle='-', linewidth=2)

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
    #plt.axis([0, max(tList)+10, 0, ymax])
    plt.title(sketch, fontsize=fontsize+1)
    plt.tight_layout(pad=0.15)

    imagename = filedir + 'boot_' + filename[7:] + '_' + norm + '.pdf'
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    #plt.show()


if __name__ == '__main__':  
    filepath = filedir + filename + '.mat'
    plot(filepath)
    
    
    
    