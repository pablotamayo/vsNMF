import tensorly as tl
from ._base_decomposition import DecompositionMixin
from collections.abc import Iterable
from .tucker_tensor import (
    tucker_to_tensor,
    TuckerTensor,
    validate_tucker_rank,
    tucker_normalize,
)
from math import sqrt
import warnings
import numpy as np
from numpy import abs, sum, where, dot, sqrt, all, max, imag, any, ones, mean, round, multiply
from .tenalg import multi_mode_dot, mode_dot
from .base import unfold
from .nnls_solvers import hals_nnls,hals_nnls_variable_sparsity, fista, active_set_nnls
from .initialize_tucker import initialize_tucker
from .projfunc import projfunc



#def non_negative_tucker_variable_sparsity(
def vsNTF(
    tensor,
    rank,
    n_iter_max=10,
    init="svd",
    tol=10e-5,
    sparsity_coefficients=None,
    normalize_initial_tensor_input = True,
    random_state=None,
    verbose=False,
    return_errors=False,
    normalize_factors=False,
):
    """
        variable-sparsity Non-Negative Tensor Factorization (vsNTF)
        T. Pham and P. Tamayo -- October 2026

        Modification of the original tensorly's Non-negative Tucker decomposition

        Iterative multiplicative update, see [2]_

    Parameters
    ----------
    tensor : ``ndarray``
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    n_iter_max : int
        maximum number of iteration
    init : {'svd', 'random'}
    sparsity_coefficients: list of len(ranks) vectors specifying the sparsity constraints for each factor matrix.
                           None entries in the list are interpret as no sparsity contraints for the corresponding factor matrices
                           if it is equal to None instead of a list, it implies no sparsity contrains for any factor matrix and
                           then the code will behave like tensorly's Non-negative Tucker decomposition
    random_state : {None, int, np.random.RandomState}
    verbose : int , optional
        level of verbosity
    return_errors : boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    normalize_factors : if True, aggregates the norms of the factors in the core.

    Returns
    -------
    core : ndarray
            positive core of the Tucker decomposition
            has shape `ranks`
    factors : ndarray list
            list of factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``

    References
    ----------
    .. [2] Yong-Deok Kim and Seungjin Choi,
       "Non-negative tucker decomposition",
       IEEE Conference on Computer Vision and Pattern Recognition s(CVPR),
       pp 1-8, 2007
    """
    rank = validate_tucker_rank(tl.shape(tensor), rank=rank)

    if isinstance(sparsity_coefficients, list):
        sparsity_constraints = True
    elif sparsity_coefficients == None:
        sparsity_constraints = False        

    epsilon = 10e-12
    if normalize_initial_tensor_input==True: #this is done to prevent overflow
        tensor = tensor/tl.max(tensor)
    # Initialisation
    nn_core, nn_factors = initialize_tucker(
        tensor,
        rank,
        range(tl.ndim(tensor)),
        init=init,
        random_state=random_state,
        non_negative=True,
    )
    mode_column_L1s_lst = []
    for mode in range(tl.ndim(tensor)):
#       if sparsity_coefficients[mode] is not None:
        if sparsity_constraints == True:
            if sparsity_coefficients[mode] is not None:
                column_L1s_lst = [] 
                side_matrix = nn_factors[mode]
                input_dimension = side_matrix.shape[0]
                side_matrix_sparsity_arr = sparsity_coefficients[mode]
                for i in range(0,len(side_matrix_sparsity_arr)):
                    column_vector_to_project = nn_factors[mode][:,i]
                    column_L1s = sqrt(input_dimension) - (sqrt(input_dimension) - 1) * side_matrix_sparsity_arr[i]
                    column_L1s_lst.append(column_L1s)
                    projected_column_vector = projfunc(column_vector_to_project, k1 = column_L1s, k2 = 1, nn = True)[0]
                    nn_factors[mode][:,i] = projected_column_vector
                mode_column_L1s_lst.append(column_L1s_lst)
            else:
                mode_column_L1s_lst.append(None)
        else:
            mode_column_L1s_lst.append(None)


    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []
    for iteration in range(n_iter_max):
        for mode in range(tl.ndim(tensor)):
            B = tucker_to_tensor((nn_core, nn_factors), skip_factor=mode)
            B = tl.transpose(unfold(B, mode))

            numerator = tl.dot(unfold(tensor, mode), B)
            numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
            denominator = tl.dot(nn_factors[mode], tl.dot(tl.transpose(B), B))
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            nn_factors[mode] *= numerator / denominator
#            if sparsity_coefficients[mode] is not None:
            if sparsity_constraints == True:
                if sparsity_coefficients[mode] is not None:
                    column_L1s_lst = mode_column_L1s_lst[mode]
                    for i in range(0,nn_factors[mode].shape[1]):
                        column_vector_to_project = nn_factors[mode][:,i]
                        factor_sparsity = sparsity_coefficients[mode][i]
                        projected_column_vector = projfunc(column_vector_to_project, k1 = column_L1s_lst[i], k2 = 1, nn = True)[0]
                        nn_factors[mode][:,i] = projected_column_vector
        numerator = tucker_to_tensor((tensor, nn_factors), transpose_factors=True)
        numerator = tl.clip(numerator, a_min=epsilon, a_max=None)
        for i, f in enumerate(nn_factors):
            if i:
                denominator = mode_dot(denominator, tl.dot(tl.transpose(f), f), i)
            else:
                denominator = mode_dot(nn_core, tl.dot(tl.transpose(f), f), i)
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
        nn_core *= numerator / denominator

        rec_error = (
            tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        )
        rec_errors.append(rec_error)
        if iteration > 1 and verbose:
            print(
                f"reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
            )

        if iteration > 1 and tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
            if verbose:
                print(f"converged in {iteration} iterations.")
            break
        if normalize_factors:
            nn_core, nn_factors = tucker_normalize((nn_core, nn_factors))
    tensor = TuckerTensor((nn_core, nn_factors))
    if return_errors:
        return tensor, rec_errors
    else:
        return tensor
