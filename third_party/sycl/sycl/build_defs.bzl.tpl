# Macros for building SYCL code.

def if_sycl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with SYCL.

    Returns a select statement which evaluates to if_true if we're building
    with SYCL enabled.  Otherwise, the select statement evaluates to if_false.
    If we are building with triSYCL instead of ComputeCPP, a list with
    the first element of if_true is returned.
    """
    return select({
        "@local_config_sycl//sycl:using_sycl_ccpp": if_true,
        "@local_config_sycl//sycl:using_sycl_trisycl": if_true[0:1],
        "//conditions:default": if_false,
    })

def if_ccpp(if_true, if_false = []):
    """Shorthand for select()'ing if we are building with ComputeCPP.

    Returns a select statement which evaluates to if_true if we're building
    with ComputeCPP enabled. Otherwise, the select statement evaluates
    to if_false.
    """
    return select({
        "@local_config_sycl//sycl:using_sycl_ccpp": if_true,
        "@local_config_sycl//sycl:using_sycl_trisycl": if_false,
        "//conditions:default": if_false,
    })
