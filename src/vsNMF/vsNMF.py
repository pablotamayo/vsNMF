from numpy.random import rand, seed
import pandas as pd
from numpy import abs, sum, where, dot, sqrt, all, max, imag, any, ones, mean, round, multiply

#log_file_path = "../projfun_log.txt"

def write_to_file(text,file_path):
    with open(file_path, 'a') as file1:
        file1.write("{}\n".format(text))
        

def projfunc(s, k1 = 0.5, k2 = 0.5, nn = True, normalize_initial_NMF_input = False,
                 save_logs = False, log_file_path="../log_file.txt"):
    
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
    if save_logs:
        write_to_file("enter while loop",log_file_path)
    while True:
        midpoint = [k1/(N - len(zerocoeff)) for i in range(N)]
        
        # This does the proposed projection operator
        if save_logs:
            write_to_file("N = {}".format(N),log_file_path)
            write_to_file("len(zerocoeff) = {}".format(len(zerocoeff)),log_file_path)
            write_to_file("Midpoint: {}".format(midpoint[0]),log_file_path)
            #print(midpoint)
            
        for i in zerocoeff:
            midpoint[i] = 0
    
        w = v - midpoint
        a = sum([x ** 2 for x in w])
        b = 2.0 * dot(w, v)    
        c = sum([x ** 2 for x in v]) - k2
        alphap = (-b + sqrt(b*b - 4*a*c))/(2*a)
        
        v = alphap * w + v
        if save_logs:
            write_to_file("w = {}".format(w),log_file_path)
            write_to_file("v = {}".format(v),log_file_path)
            write_to_file("a = {}".format(a),log_file_path)
            write_to_file("b = {}".format(b),log_file_path)
            write_to_file("c = {}".format(c),log_file_path)
            write_to_file("b*b = {}".format(b*b),log_file_path)
            write_to_file("4ac = {}".format(4*a*c),log_file_path)
            write_to_file("alphap = {}".format(alphap),log_file_path)
            write_to_file("len(v) = {}".format(len(v)),log_file_path)

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

def projfunc_original(s, k1 = 0.5, k2 = 0.5, nn = True):
    
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

    while True:

        # This does the proposed projection operator
        midpoint = [k1/(N - len(zerocoeff)) for i in range(N)]
    
        for i in zerocoeff:
            midpoint[i] = 0
    
        w = v - midpoint
        a = sum([x ** 2 for x in w])
        b = 2.0 * dot(w, v)    
        c = sum([x ** 2 for x in v]) - k2
        alphap = (-b + sqrt(b*b - 4*a*c))/(2*a)
        v = alphap * w + v

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

def sparseness(A, matrix_type):
    # Computes the average (columns/row) sparseness of a NMFSC W or H matrix according to:
    # sparseness(x) = (sqrt(n) - (sum(abs(x_i)))/sqrt(sum(x_i)^2))/(sqrt(n) - 1)
    #
    # Original paper: P. Hoyer, 'Non-negative matrix factorization with sparseness constraints'
    # Jour. of Machine Learning Research 5:1457-1469, 2004. https://arxiv.org/abs/cs/0408058
    # Python version by P. Tamayo Nov 5, 2020
    # INPUTS:
    # A          - a W or H NMFSC matrix 
    # matrix_type - type of matrix 'W' or 'H'
    
    if matrix_type == 'W':
        s_vec = []
        n = A.shape[0]
        for i in range(A.shape[1]):
            s = (sqrt(n) - A.iloc[:, i].abs().sum()/sqrt(A.iloc[:, i].pow(2.0).sum()))/(sqrt(n) - 1)
            s_vec.append(s)
        aver_sparseness = mean(s_vec)
        
    elif matrix_type == 'H':
        s_vec = []
        n = A.shape[1]
        for i in range(A.shape[0]):
            s = (sqrt(n) - A.iloc[i, :].abs().sum()/sqrt(A.iloc[i, :].pow(2.0).sum()))/(sqrt(n) - 1)
            s_vec.append(s)
        aver_sparseness = mean(s_vec)
    else:
        raise('ERROR: unknown matrix time: {}'.format(matrix_type))
        
    return aver_sparseness

def compute_sparseness(W, H):
    # Computes the average (columns/row) sparseness of an NMF W and H matrices according to:
    # sparseness(x) = (sqrt(n) - (sum(abs(x_i)))/sqrt(sum(x_i)^2))/(sqrt(n) - 1)
    #
    # Original paper: P. Hoyer, 'Non-negative matrix factorization with sparseness constraints'
    # Jour. of Machine Learning Research 5:1457-1469, 2004. https://arxiv.org/abs/cs/0408058
    # Python version by P. Tamayo Nov 5, 2020
    # INPUTS:
    # W and H NMF matrices 
    
    s_w_vec = pd.Series(0.0, index = W.columns, name = 's w vec')
    n = W.shape[0]
    for c in W.columns:
        s = (sqrt(n) - W.loc[:, c].abs().sum()/sqrt(W.loc[:, c].pow(2.0).sum()))/(sqrt(n) - 1)
        s_w_vec.loc[c] = s
        
    w_aver_sparseness = s_w_vec.mean()

    s_h_vec = pd.Series(0.0, index = H.index, name = 's h vec')        
    n = H.shape[1]
    for r in H.index:
        s = (sqrt(n) - H.loc[r, :].abs().sum()/sqrt(H.loc[r, :].pow(2.0).sum()))/(sqrt(n) - 1)
        s_h_vec.loc[r] = s
        
    h_aver_sparseness = s_h_vec.mean()

    s_w_vec.sort_values(ascending = False, inplace = True)
    #print('W columns')
    #print(s_w_vec)
    
    s_h_vec.sort_values(ascending = False, inplace = True)
    #print('H rows')
    #print(s_h_vec)

    return s_w_vec, w_aver_sparseness, s_h_vec, h_aver_sparseness


def compute_row_sparseness(G):
    # Computes the average (row) sparseness of an NMF H or W matrices, or NTF factor matrix,
    # according to:
    # sparseness(x) = (sqrt(n) - (sum(abs(x_i)))/sqrt(sum(x_i)^2))/(sqrt(n) - 1)
    #
    # Original paper: P. Hoyer, 'Non-negative matrix factorization with sparseness constraints'
    # Jour. of Machine Learning Research 5:1457-1469, 2004. https://arxiv.org/abs/cs/0408058
    # Python version by P. Tamayo Nov 5, 2020
    # INPUTS:
    # G NMF or NTF matrix 
    
    s_G_vec = pd.Series(0.0, index = G.index, name = 's G vec')
    n = G.shape[1]
    for r in G.index:
        s = (sqrt(n) - G.loc[r, :].abs().sum()/sqrt(G.loc[r, :].pow(2.0).sum()))/(sqrt(n) - 1)
        s_G_vec.loc[r] = s
        
    G_aver_sparseness = s_G_vec.mean()

    return s_G_vec, G_aver_sparseness


def vsNMF(V,
          rdim,
          niter,
          sW ,
          sH ,
          initW = None,
          initH = None,
          random_seed = 17895314,
          normalize_initial_NMF_input = True,
          save_logs = False,
          verbose = True,
          log_file_path = "../log_file.txt"):
    #
    # Non-negative matrix factorization with variable sparseness constraints
    #
    # INPUTS:
    # V          - data matrix 
    # rdim       - number of components (inner dimension of factorization)
    # sW         - sparseness of W, in [0,1]. (give [] if no constraint)
    # sH         - sparseness of H, in [0,1]. (give [] if no constraint)
    # niter      - number of iterations
    # initW      - initial W matrix
    # initH      - initial H matrix
    # random_seed - seed for random number generator
    #
    # Note: Sparseness is measured on the scale [0,1] where 0 means
    # completely distributed and 1 means ultimate sparseness.
    #
    # Original MATLAB code Written 2.7.2004 by Patrik O. Hoyer (https://github.com/aludnam/MATLAB/tree/master/nmfpack)
    # Original paper: P. Hoyer, 'Non-negative matrix factorization with sparseness constraints'
    # Jour. of Machine Learning Research 5:1457-1469, 2004. https://arxiv.org/abs/cs/0408058
    # Python version with some modifications by P. Tamayo Nov 5, 2020

    seed(random_seed)
    
    # Check that we have non-negative data
    if any(V < 0):
        raise('Error: negative values in input matrix V')
    
    # Globally rescale data to avoid potential overflow/underflow
    if normalize_initial_NMF_input==True:
        V = V/V.max().max()

    # Data dimensions
    vdim = V.shape[0]
    samples = V.shape[1]
    
    # Create initial matrices
    if initW is None:
        W = pd.DataFrame(rand(vdim, rdim), index = V.index, # ['k{}'.format(r) for r in range(vdim)], 
                         columns= ['F{}'.format(r) for r in range(rdim)])
    else:
        W = initW

    if initH is None:
        H = pd.DataFrame(rand(rdim, samples), index = ['F{}'.format(r) for r in range(rdim)], 
                         columns = V.columns) # ['c{}'.format(r) for r in range(samples)])
    else:
        H = initH
        
    ones_df = pd.DataFrame(ones((1, samples)), index = [0], columns = V.columns) # ['c{}'.format(r) for r in range(samples)])
    H = H.div(sum(H*H, axis = 1).to_frame().pow(0.5).dot(ones_df))

    if verbose:
        print('Initial W sparseness: {}'.format(round(sparseness(W, matrix_type = 'W'), 3)))
        print('Initial H sparseness: {}'.format(round(sparseness(H, matrix_type = 'H'), 3)))

    # Make initial matrices have correct sparseness
    if not len(sW) == 0.0:
        row_L1a_lst = []
        #L1a = sqrt(vdim) - (sqrt(vdim) - 1) * sW
        for i in range(rdim):
            row_L1a = sqrt(vdim) - (sqrt(vdim) - 1) * sW[i]
            row_L1a_lst.append(row_L1a)
            W.iloc[:, i] = list(projfunc(W.iloc[:, i].values, row_L1a, 1, 1)[0])

    if not len(sH) == 0.0:
        row_L1s_lst = []
        #L1s = sqrt(samples) - (sqrt(samples) - 1) * sH
        for i in range(rdim):
            row_L1s = sqrt(samples) - (sqrt(samples) - 1) * sH[i]
            row_L1s_lst.append(row_L1s)
            H.iloc[i, :] = list(projfunc(H.iloc[i, :].values, row_L1s, 1, 1)[0])
                
    
    # Calculate initial objective
    objhistory = [0.5 * (V - W.dot(H)).pow(2.0).values.sum()]

    # Initial stepsizes
    stepsizeW = 1
    stepsizeH = 1

    # Start iterations
    if verbose:
        print('Running NMF with Variable Sparse Constraints ({} iterations)'.format(niter))
        print('\n iteration: 0 (0%)', end=' ')
    #print(row_L1s_lst)
    for iter in range(niter):

        if iter % 5 == 0:
            print('\r iteration: {} ({}%)'.format(iter, round(100*iter/niter, 3)), end=' ') 

        # Save old values
        Wold = W
        Hold = H
        
        # ----- Update H ---------------------------------------
    
        if not len(sH) == 0.0:
            # Gradient for H
            dH = W.T.dot(W.dot(H).sub(V))
            if save_logs:
                write_to_file("Iteration {} dH".format(iter),log_file_path)
                write_to_file(dH,log_file_path)
            begobj = objhistory[-1]
        
            # Make sure we decrease the objective!
            while True:
                # Take step in direction of negative gradient, and project
                Hnew = H - stepsizeH * dH
                for i in range(rdim):
                    #print(i)
                    if save_logs:
                        write_to_file("Calling projfunc for H with len(s) = {}".format(len(Hnew.iloc[i, :].values)),log_file_path)
                        write_to_file("Sum = {}".format(Hnew.iloc[i, :].values.sum()),log_file_path)
                    row_L1s = row_L1s_lst[i]
                    Hnew.iloc[i, :] = list(projfunc(Hnew.iloc[i, :].values, k1 = row_L1s, k2 = 1, nn = 1, save_logs = False,
                                                        log_file_path  =log_file_path)[0])
                    
                # Calculate new objective
                #print("Hnew: {}".format(Hnew.shape))
                newobj = 0.5 * (V - W.dot(Hnew)).pow(2.0).values.sum()
                #print("newobj: {}".format(newobj))
                #print("begobj: {}".format(begobj))
                # If the objective decreased, we can continue...
                if newobj <= begobj:
                    break
                else:
                    stepsizeH = stepsizeH/2
                    if stepsizeH < 1e-200:
                        #print('Algorithm converged')
                        return (W, H, objhistory)
        
            # Slightly increase the stepsize
            stepsizeH = stepsizeH * 1.2
            H = Hnew

        else:
            # Update using standard NMF multiplicative update rule
            
            dH = W.T.dot(W.dot(H).sub(V))
            if save_logs:
                write_to_file("Iteration {} dH".format(iter),log_file_path)
                write_to_file(dH,log_file_path)
            #H = H * W.T.dot(V).div(W.T.dot(W).dot(H) + 1e-9)
            H = H * W.T.dot(V).div(W.T.dot(W).dot(H) + 1e-9)

            # Renormalize so rows of H have constant energy
            #norms = sum(H.pow(2.0), axis = 1).to_frame().pow(0.5)
            norms = sum(H.pow(2.0), axis = 1).to_frame().pow(0.5)
            #H = H.div(norms.dot(ones_df))
            H = H.div(norms.dot(ones_df))
            #ones_df2 = pd.DataFrame(ones((vdim, 1)), index = V.index, columns = [0]) # ['r{}'.format(r) for r in range(vdim)]
            ones_df2 = pd.DataFrame(ones((vdim, 1)), index = V.index, columns = [0])
            #W = W.div(ones_df2.dot(norms))
            W = multiply(W , ones_df2.dot(norms.T))

        # ----- Update W ---------------------------------------

        # Gradient for H
        if not len(sW) == 0.0:
    
            # Gradient for W
            dW = (W.dot(H) - V).dot(H.T)
            if save_logs:
                write_to_file("Iteration {} dW".format(iter),log_file_path)
                write_to_file(dW,log_file_path)
            begobj = 0.5 * (V - W.dot(H)).pow(2.0).values.sum()

            # Make sure we decrease the objective!
            while True:
                # Take step in direction of negative gradient, and project
                Wnew = W - stepsizeW * dW
                norms = sum(Wnew.pow(2.0), axis = 0).to_frame().pow(0.5)

                for i in range(rdim):
                    if save_logs:
                        write_to_file("Calling projfunc for W with len(s) = {}".format(len(Wnew.iloc[:, i].values)),log_file_path)
                        write_to_file("W input: sum=".format(sum(Wnew.iloc[:, i].values)),log_file_path)
                        write_to_file(Wnew.iloc[:, i].values,log_file_path)
                    row_L1a = row_L1a_lst[i]
                    Wnew.iloc[:, i] = list(projfunc(Wnew.iloc[:, i].values, k1=row_L1a * norms.iloc[i, 0], k2=norms.iloc[i, 0] * norms.iloc[i, 0], nn=1)[0])

                # Calculate new objective
                newobj = 0.5 * (V - Wnew.dot(H)).pow(2.0).values.sum()
                # If the objective decreased, we can continue...

                if newobj <= begobj:
                    break

                # else decrease stepsize and try again
                else:
                    stepsizeW = stepsizeW/2.0

                    if stepsizeW < 1e-200:
                        #print('Algorithm converged.')
                        return (W, H, objhistory)

            # Slightly increase the stepsize
            stepsizeW = stepsizeW * 1.2
            W = Wnew

        else:
        
            # Update using standard NMF multiplicative update rule
        
            W = W * V.dot(H.T).div(W.dot(H).dot(H.T) + 1e-9)

        # Calculate objective
        newobj = 0.5 * (V - W.dot(H)).pow(2.0).values.sum()
        objhistory.append(newobj)

    if verbose:
        print('\n Final W sparseness: {}'.format(round(sparseness(W, matrix_type = 'W'), 3)))
        print(' x Final H sparseness: {}'.format(round(sparseness(H, matrix_type = 'H'), 3)))

    #compute_sparseness(W, H)
        
    return (W, H, objhistory)
