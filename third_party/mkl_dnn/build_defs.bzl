def if_mkl_open_source_only(if_true, if_false = []):
    """Returns `if_true` if MKL-DNN v0.x is used.

    Shorthand for select()'ing on whether we're building with
    MKL-DNN v0.x open source library only, without depending on MKL binary form.

    Returns a select statement which evaluates to if_true if we're building
    with MKL-DNN v0.x open source library only. Otherwise, the select statement
    evaluates to if_false.

    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkl_opensource": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_openmp(if_true, if_false = []):
    """Returns `if_true` if OpenMP is used with oneDNN.

    Shorthand for select()'ing on whether we're building with
    oneDNN open source library only with openmp

    Returns a select statement which evaluates to if_true if we're building
    with oneDNN open source library only with OpenMP. Otherwise, the
    select statement evaluates to if_false.

    """
    return select({
        "@org_tensorflow//third_party/mkl_dnn:build_with_mkldnn_openmp": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_aarch64_acl(if_true, if_false = []):
    return select({
        "@org_tensorflow//third_party/mkl:build_with_mkl_aarch64": if_true,
        "//conditions:default": if_false,
    })
