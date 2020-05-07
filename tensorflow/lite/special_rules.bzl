"""External versions of build rules that differ outside of Google."""

load(
    "//tensorflow:tensorflow.bzl",
    "clean_dep",
)

def tflite_portable_test_suite(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def tflite_portable_test_suite_combined(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def tflite_ios_per_kernel_test(**kwargs):
    """This is a no-op outside of Google."""
    _ignore = [kwargs]
    pass

def ios_visibility_whitelist():
    """This is a no-op outside of Google."""
    pass

def tflite_extra_gles_deps():
    """This is a no-op outside of Google."""
    return []

def tflite_ios_lab_runner(version):
    """This is a no-op outside of Google."""

    # Can switch back to None when https://github.com/bazelbuild/rules_apple/pull/757 is fixed
    return "@build_bazel_rules_apple//apple/testing/default_runner:ios_default_runner"

def if_nnapi(supported, not_supported = [], supported_android = None):
    if supported_android == None:
        supported_android = supported

    # We use a blacklist rather than a whitelist for known unsupported platforms.
    return select({
        clean_dep("//tensorflow:emscripten"): not_supported,
        clean_dep("//tensorflow:ios"): not_supported,
        clean_dep("//tensorflow:macos"): not_supported,
        clean_dep("//tensorflow:windows"): not_supported,
        clean_dep("//tensorflow:android"): supported_android,
        "//conditions:default": supported,
    })
