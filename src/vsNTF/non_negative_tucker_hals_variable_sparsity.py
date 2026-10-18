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

def non_negative_tucker_hals_variable_sparsity(
    tensor,
    rank,
    n_iter_max=100,
    init="svd",
    svd="truncated_svd",
    tol=1e-8,
    sparsity_coefficients=None,
    core_sparsity_coefficient=None,
    fixed_modes=None,
    random_state=None,
    verbose=False,
    normalize_factors=False,
    return_errors=False,
    exact=False,
    algorithm="fista",
):
    r"""Non-negative Tucker decomposition with HALS

    Uses HALS to update each factor columnwise and uses
    fista or active set algorithm to update the core, see [1]_ 
    
    Parameters
    ----------
    tensor : ndarray
    rank : None, int or int list
        size of the core tensor, ``(len(ranks) == tensor.ndim)``
        if int, the same rank is used for all modes
    n_iter_max : int
            maximum number of iteration
    init : {'svd', 'random'}, optional
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    tol : float, optional
        tolerance: the algorithm stops when the variation in
        the reconstruction error is less than the tolerance
        Default: 1e-8
    sparsity_coefficients : array of array of float (as much as the number of modes)
        The sparsity coefficients are used for each factor
        If set to None, the algorithm is computed without sparsity
        Default: None
    core_sparsity_coefficient : array of float. This coefficient imposes sparsity on core
        when it is updated with fista.
        Default: None
    fixed_modes : array of integers (between 0 and the number of modes)
        Has to be set not to update a factor, 0 and 1 for U and V respectively
        Default: None
    verbose : boolean
        Indicates whether the algorithm prints the successive
        reconstruction errors or not
        Default: False
    normalize_factors : if True, aggregates the norms of the factors in the core.
    return_errors : boolean
        Indicates whether the algorithm should return all reconstruction errors
        and computation time of each iteration or not
        Default: False
    exact : If it is True, the HALS nnls subroutines give results with high precision but it has a higher computational cost.
        If it is False, the algorithm gives an approximate solution.
        Default: False
    algorithm : {'fista', 'active_set'}
        Non negative least square solution to update the core. 
        Default: 'fista'

    Returns
    -------
    factors : ndarray list
            list of positive factors of the CP decomposition
            element `i` is of shape ``(tensor.shape[i], rank)``
    errors: list
        A list of reconstruction errors at each iteration of the algorithm.

    Notes
    -----
    Tucker decomposes a tensor into a core tensor and list of factors:

    .. math::

            tensor = [| core; factors[0], ... ,factors[-1] |],

    We solve the following problem for each factor:

    .. math::

            \min_{tensor >= 0} ||tensor_i - factors[i]\times core_i \times (\prod_{i\neq j}(factors[j]))^T||^2,

    If we define two variables such as:

    .. math::

            U = core_i \times (\prod_{i \neq j}(factors[j] \times factors[j]^T)), \\
            M = tensor_i,

    Gradient of the problem becomes:

    .. math::

            \delta = -U^TM + factors[i] \times U^TU,

    In order to calculate UTU and UTM, we define two variables:

    .. math::

            CoreCross = \prod_{i\neq j}(core_i \times (\prod_{i\neq j}(factors[j]\times factors[j]^T)) \\
            TensorCross =  \prod_{i\neq j} tensor_i \times factors[i],

    Then UTU and UTM becomes:

    .. math::

            U^TU = CoreCross_j \times core_j^T, \\
            U^TM = (TensorCross_j \times core_j^T)^T,

    References
    ----------
    .. [1] G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    rank = validate_tucker_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)
    if sparsity_coefficients is None or not isinstance(sparsity_coefficients, Iterable):
        sparsity_coefficients = [sparsity_coefficients] * n_modes

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn(
            "You asked for fixing the last mode, which is not supported. The last mode will not be fixed."
            " Consider using tl.moveaxis() to permute it to another position and keep it fixed there."
        )
        fixed_modes.remove(tl.ndim(tensor) - 1)

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None
    # Generating the mode update sequence
    modes = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    nn_core, nn_factors = initialize_tucker(
        tensor,
        rank,
        modes,
        init=init,
        svd=svd,
        random_state=random_state,
        non_negative=True,
    )
    # initialisation - declare local variables
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        for mode in modes:
            # Computing Hadamard of cross-products
            pseudo_inverse = nn_factors.copy()
            for i, factor in enumerate(nn_factors):
                if i != mode:
                    pseudo_inverse[i] = tl.dot(tl.conj(tl.transpose(factor)), factor)
            # UtU
            core_cross = multi_mode_dot(nn_core, pseudo_inverse, skip=mode)
            UtU = tl.dot(unfold(core_cross, mode), tl.transpose(unfold(nn_core, mode)))

            # UtM
            tensor_cross = multi_mode_dot(tensor, nn_factors, skip=mode, transpose=True)
            MtU = tl.dot(
                unfold(tensor_cross, mode), tl.transpose(unfold(nn_core, mode))
            )
            UtM = tl.transpose(MtU)

            # Call the hals resolution with nnls, optimizing the current mode
            nn_factor = hals_nnls_variable_sparsity(
                UtM,
                UtU,
                tl.transpose(nn_factors[mode]),
                n_iter_max=100,
                sparsity_coefficient_arr=sparsity_coefficients[mode],
                exact=exact,
            )
            nn_factors[mode] = tl.transpose(nn_factor)
        # updating core
        if algorithm == "fista":
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_estimation = multi_mode_dot(tensor, nn_factors, transpose=True)
            learning_rate = 1

            for MtM in pseudo_inverse:
                learning_rate *= 1 / (tl.truncated_svd(MtM)[1][0])
            nn_core = fista(
                core_estimation,
                pseudo_inverse,
                x=nn_core,
                n_iter_max=n_iter_max,
                sparsity_coef=core_sparsity_coefficient,
                lr=learning_rate,
            )
        if algorithm == "active_set":
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_estimation_vec = tl.base.tensor_to_vec(
                tl.tenalg.mode_dot(
                    tensor_cross, tl.transpose(nn_factors[modes[-1]]), modes[-1]
                )
            )
            pseudo_inverse_kr = tl.tenalg.kronecker(pseudo_inverse)
            vectorcore = active_set_nnls(
                core_estimation_vec, pseudo_inverse_kr, x=nn_core, n_iter_max=n_iter_max
            )
            nn_core = tl.reshape(vectorcore, tl.shape(nn_core))

        # Adding the l1 norm value to the reconstruction error
        sparsity_error = 0
        for index, sparse in enumerate(sparsity_coefficients):
            if sparse is not None: 
                l1_norm_columns = []
                for i in range(0,len(sparse)):
                    l1_norm_columns.append(sparse[i] * tl.norm(nn_factors[index][i], order=1))
                sparsity_error += 2 * tl.max(np.array(l1_norm_columns))
                #sparsity_error += 2 * (sparse * tl.norm(nn_factors[index], order=1))
        # error computation
        rec_error = (
            tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        )
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(
                    f"reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
                )

            if tol and tl.abs(rec_errors[-2] - rec_errors[-1]) < tol:
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

def initialize_tucker(
    tensor,
    rank,
    modes,
    random_state,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
    mask=None,
    svd_mask_repeats=5,
):
    """
    Initialize core and factors used in `tucker`.
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.

    Parameters
    ----------
    tensor : ndarray
    rank : int
           number of components
    modes : int list
    random_state : {None, int, np.random.RandomState}
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'truncated_svd'
          function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned

    Returns
    -------
    core    : ndarray
              initialized core tensor
    factors : list of factors
    """
    # Initialisation
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            mask_unfold = None if mask is None else unfold(mask, mode)
            U, _, _ = svd_interface(
                unfold(tensor, mode),
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                random_state=random_state,
            )

            factors.append(U)
        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

    elif init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample([rank[index] for index in range(len(modes))]) + 0.01,
            **tl.context(tensor),
        )  # Check this
        factors = [
            tl.tensor(
                rng.random_sample((tensor.shape[mode], rank[index])),
                **tl.context(tensor),
            )
            for index, mode in enumerate(modes)
        ]

    else:
        (core, factors) = init

    if non_negative is True:
        factors = [tl.abs(f) for f in factors]
        core = tl.abs(core)

    return core, factors
