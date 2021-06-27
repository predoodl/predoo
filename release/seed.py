import numpy as np

def seed_mutation(x, corpus):
    a = []
    a1 = 0.0001 * np.ones(x.shape, np.float64)
    a2 = 0.000001 * np.ones(x.shape, np.float64)
    a3 = 0.00000001 * np.ones(x.shape, np.float64)
    corpus.put(x + a1)
    corpus.put(x + a2)
    corpus.put(x + a3)