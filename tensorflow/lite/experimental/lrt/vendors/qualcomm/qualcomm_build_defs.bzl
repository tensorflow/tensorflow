"""Build definitions for QualComm backend."""

load("//tensorflow/lite/experimental/lrt/build_common:lite_rt_build_defs.bzl", "append_rule_kwargs", "lite_rt_lib", "make_rpaths")

_QNN_LIBCC = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt:lib/x86_64-linux-clang/libc++.so.1",
    # "//third_party/qairt:lib/x86_64-linux-clang/libc++abi.so.1",
    # copybara:uncomment_end
]  # @unused

# TODO: Make rpaths dynamic with "$(location {})".
_QNN_LIB_RPATHS = [
    # copybara:uncomment_begin(google-only)
    # "third_party/qairt/lib/x86_64-linux-clang",
    # copybara:uncomment_end
]

_QNN_LIB_HTP = [
    # copybara:uncomment_begin(google-only)
    # "//third_party/qairt:lib/x86_64-linux-clang/libQnnHtp.so",
    # copybara:uncomment_end
]

def lite_rt_lib_with_qnn(
        backend = "htp",
        use_custom_libcc = False,
        **lite_rt_lib_kwargs):
    """Creates a lite_rt_lib target with QualComm backend dependencies.

    Args:
        backend: The backend to use. Currently only "htp" is supported.
        use_custom_libcc: Whether to use a custom libcc. Not yet supported.
        **lite_rt_lib_kwargs: Keyword arguments passed to lite_rt_lib.
    """
    if backend != "htp":
        fail("Only htp currently supported")

    if use_custom_libcc:
        # TODO: Figure out strategy for custom libcc.
        fail("Custom libcc not yet supported")

    append_rule_kwargs(
        lite_rt_lib_kwargs,
        data = [_QNN_LIB_HTP],
        linkopts = [make_rpaths(_QNN_LIB_RPATHS)],
    )

    lite_rt_lib(**lite_rt_lib_kwargs)
