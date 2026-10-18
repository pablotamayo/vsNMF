import sys
from .base import unfold, fold
from .base import vec_to_tensor

from .backend import (
    set_backend,
    get_backend,
    # backend_context,
    # backend_manager,
    #    _get_backend_dir, _get_backend_method,
)

# from . import backend as backend_manager
from .backend import (
    context,
    tensor,
    is_tensor,
    shape,
    ndim,
    to_numpy,
    copy,
    float64,
    concatenate,
    reshape,
    transpose,
    moveaxis,
    arange,
    any,
    ones,
    zeros,
    zeros_like,
    eye,
    where,
    conj,
    index,
    index_update,
    clip,
    max,
    min,
    argmax,
    argmin,
    all,
    mean,
    sum,
    prod,
    sign,
    abs,
    sqrt,
    norm,
    dot,
    kron,
    einsum,
    solve,
    lstsq,
    qr,
    kr,
    stack,
    maximum,
    eps,
    finfo,
    matmul,
    index_update,
    check_random_state,
    randn,
    log,
    log2,
    exp,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    arcsin,
    arccos,
    arctan,
    sinh,
    sort,
    cosh,
    tanh,
    arcsinh,
    arccosh,
    arctanh,
    asinh,
    acosh,
    atanh,
    e,
    pi,
    inf,
    nan,
    int32,
    int64,
)

from . import backend


# Add Backend functions, dynamically dispatched
def __dir__():
    """Returns the module's __dir__, including the local variables
    and augmenting it with the dynamically dispatched variables from backend.
    """
    static_items = list(sys.modules[__name__].__dict__.keys())
    return backend.get_backend_dir() + static_items
    # return _get_backend_dir() + static_items


__getattr__ = backend.__getattribute__
from .non_negative_tucker_hals_variable_sparsity import non_negative_tucker_hals_variable_sparsity
#from .non_negative_tucker_variable_sparsity import non_negative_tucker_variable_sparsity
from .vsNTF import vsNTF
