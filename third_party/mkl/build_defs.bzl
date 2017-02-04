# Macros for building MKL code.

def if_mkl(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with MKL.

    Returns a select statement which evaluates to if_true if we're building
    with MKL enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "//third_party/mkl:using_mkl": if_true,
        "//conditions:default": if_false
    })
