import numpy as np

# If cupy is available, import it and use it instead of numpy.
# Use `xp` as the interface to numpy or cupy.
def import_cupy_xp():
    import imp
    cp = None
    xp = np
    try:
        imp.find_module('cupy')
        import cupy as cp
        xp = cp
    except ImportError:
        None
    return cp, xp

cp, xp = import_cupy_xp()

# Convert a numpy or cupy array to numpy array/scalar.
# This is useful when returning values to the user.
# TODO: Accept type `list`?
def to_numpy(x):
    out = x
    # Convert to a numpy array
    if xp == cp and type(x) == cp.ndarray:
        out = x.get()
    # Make this an array, if not already
    out = np.array(out)
    # Make arrays of shape () a scalar
    if out.shape == ():
        out = np.asscalar(out)
    return out

def is_positive_semi_definite(R):
    if not isinstance(R, (xp.ndarray, xp.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return xp.all(xp.linalg.eigvals(R) > 0)
