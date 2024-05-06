"""Starlark macros for oneDNN.

if_mkldnn_openmp checks if we are building x86 backend with OpenMP.
if_mkldnn_aarch64_acl checks if we are building with Arm Compute Library.
if_mkldnn_aarch64_acl_openmp checks if we are building ACL with OpenMP.
"""

def if_mkldnn_openmp(if_true, if_false = []):
    """Returns `if_true` if OpenMP is used with oneDNN.

    Shorthand for select()'ing on whether we're building with
    oneDNN open source library only with openmp

    Returns a select statement which evaluates to if_true if we're building
    with oneDNN open source library only with OpenMP. Otherwise, the
    select statement evaluates to if_false.

    """
    return select({
        "@local_tsl//third_party/mkl_dnn:build_with_mkldnn_openmp": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_aarch64_acl(if_true, if_false = []):
    return select({
        "@local_xla//xla/tsl/mkl:build_with_mkl_aarch64": if_true,
        "//conditions:default": if_false,
    })

def if_mkldnn_aarch64_acl_openmp(if_true, if_false = []):
    return select({
        "@local_tsl//third_party/mkl_dnn:build_with_mkl_aarch64_openmp": if_true,
        "//conditions:default": if_false,
    })

def gen_onednn_version(name, header_in, header_out, **kwargs):
    tool = "@org_tensorflow//third_party/mkl_dnn:gen_onednn_version"

    native.genrule(
        name = name,
        srcs = [header_in],
        outs = [header_out],
        tools = [tool],
        cmd = "$(location {}) ".format(tool) + "--in=$< " + "--out=$@",
        **kwargs
    )
