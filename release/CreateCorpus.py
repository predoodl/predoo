import numpy as np
from queue import Queue

def createCorpus(size,shape):
    q = Queue()
    for i in range(size):
        if len(shape)==4:
            x = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        else:
            x = np.random.randn(shape[0],shape[1])
        q.put(x)
    return q



if __name__=='__main__':
    shape= [10, 10]
    createCorpus(1,shape)