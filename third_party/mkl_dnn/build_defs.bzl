def if_mkl_open_source_only(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with
    MKL-DNN open source lib only, without depending on MKL binary form.

    Returns a select statement which evaluates to if_true if we're building
    with MKL-DNN open source lib only. Otherwise,
    the select statement evaluates to if_false.

    """
    return select({
        str(Label("//third_party/mkl_dnn:using_mkl_dnn_only")): if_true,
        "//conditions:default": if_false,
    })
