# Macros for building CUDA code.

def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA.

    Returns a select statement which evaluates to if_true if we're building
    with CUDA enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_gcudacc": if_true,
        "//conditions:default": if_false
    })
