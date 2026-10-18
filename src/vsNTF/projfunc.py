import numpy as np
from numpy import abs, sum, where, dot, sqrt, all, max, imag, any, ones, mean, round, multiply

def projfunc(s, k1 = 0.5, k2 = 0.5, nn = True):
    
    # Solves the following problem:
    # Given a vector s, find the vector v having sum(abs(v))=k1 
    # and sum(v.^2)=k2 which is closest to s in the euclidian sense.
    # If the binary flag nn is set, the vector v is additionally
    # restricted to being non-negative (v>=0).
    #
    # Original MATLAB code Written 2.7.2004 by Patrik O. Hoyer (https://github.com/aludnam/MATLAB/tree/master/nmfpack)
    # Original paper: P. Hoyer, 'Non-negative matrix factorization with sparseness constraints'
    # Jour. of Machine Learning Research 5:1457-1469, 2004. https://arxiv.org/abs/cs/0408058
    # Python version by P. Tamayo Nov 5, 2020

    # Problem dimension
    N = len(s)

    # If non-negativity flag not set, record signs and take abs
    if nn == False:
        isneg = [1 if x < 0 else 0 for x in s]
        s = list(abs(s))

    # Start by projecting the point to the sum constraint hyperplane
    v = s + (k1 - sum(s))/N
    
    # Initialize zerocoeff (initially, no elements are assumed zero)
    zerocoeff = []
    j = 0
    #print("test projection",flush=True)
    #log_folder = "/home/computer/sparse_tensor_factorization/logs"
    while True:

        # This does the proposed projection operator
        midpoint = [k1/(N - len(zerocoeff)) for i in range(N)]
    
        for i in zerocoeff:
            midpoint[i] = 0
    
        w = v - midpoint
        a = sum([x ** 2 for x in w])
        b = 2.0 * dot(w, v)    
        c = sum([x ** 2 for x in v]) - k2
        #write_to_file("a = {}".format(a), join(log_folder,"projection_log.txt"))
        #write_to_file("b = {}".format(b), join(log_folder,"projection_log.txt"))
        #write_to_file("c = {}".format(c), join(log_folder,"projection_log.txt"))
        #write_to_file("b*b = {}".format(b*b), join(log_folder,"projection_log.txt"))
        #write_to_file("4*a*c = {}".format(4*a*c), join(log_folder,"projection_log.txt"))
        alphap = (-b + sqrt(b*b - 4*a*c))/(2*a)
        v = alphap * w + v
        #write_to_file("alphap = {}".format(alphap), join(log_folder,"projection_log.txt"))
        #write_to_file("v = {}".format(v), join(log_folder,"projection_log.txt"))
        
        # print('v={}'.format(v), flush=True)
        
        if all(v >= 0):
        # We've found our solution
            usediters = j + 1;
            break

        j = j + 1
        
        # Set negs to zero, subtract appropriate amount from rest

        zerocoeff = []
        for i in range(len(v)):
            if v[i] <= 0:
                v[i] = 0
                zerocoeff.append(i)
                
        tempsum = sum(v)
        v = v + (k1 - tempsum)/(N - len(zerocoeff))
        
        for i in zerocoeff:
            v[i] = 0

    # If non-negativity flag not set, return signs to solution
    if (nn == False):
        isneg_c = [(-2 * x + 1) for x in isneg]
        v = [x * y for x, y in zip(v, isneg_c)]

    # Check for problems
    if max(abs(imag(v))) > 1e-10:
        raise('Somehow got imaginary values!')
    
    return (list(v), usediters)
