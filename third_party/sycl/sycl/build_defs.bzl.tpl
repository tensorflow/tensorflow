# Macros for building SYCL code.

def if_sycl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with SYCL.

    Returns a select statement which evaluates to if_true if we're building
    with SYCL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_sycl//sycl:using_sycl": if_true,
        "//conditions:default": if_false
    })
