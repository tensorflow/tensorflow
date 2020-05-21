def if_mkl_open_source_only(if_true, if_false = []):
    """Returns `if_true` if MKL-DNN v0.x is used.

    Shorthand for select()'ing on whether we're building with
    MKL-DNN v0.x open source library only, without depending on MKL binary form.

    Returns a select statement which evaluates to if_true if we're building
    with MKL-DNN v0.x open source library only. Otherwise, the select statement
    evaluates to if_false.

    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkl_dnn_only": if_true,
        "//conditions:default": if_false,
    })

def if_mkl_v1_open_source_only(if_true, if_false = []):
    """Returns `if_true` if MKL-DNN v1.x is used.

    Shorthand for select()'ing on whether we're building with
    MKL-DNN v1.x open source library only, without depending on MKL binary form.

    Returns a select statement which evaluates to if_true if we're building
    with MKL-DNN v1.x open source library only. Otherwise, the
    select statement evaluates to if_false.

    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkl_dnn_v1_only": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_threadpool(if_true, if_false = []):
    """Returns `if_true` if MKL-DNN v1.x is used.

    Shorthand for select()'ing on whether we're building with
    MKL-DNN v1.x open source library only with user specified threadpool, without depending on MKL binary form.

    Returns a select statement which evaluates to if_true if we're building
    with MKL-DNN v1.x open source library only with user specified threadpool. Otherwise, the
    select statement evaluates to if_false.

    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkldnn_threadpool": if_true,
        "//conditions:default": if_false,
    })
