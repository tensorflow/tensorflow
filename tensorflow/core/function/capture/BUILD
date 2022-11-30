load("//tensorflow:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/core/function/capture:__subpackages__",
        "//tensorflow/python/eager/polymorphic_function:__subpackages__",
    ],
)

py_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
    ],
)

py_strict_test(
    name = "free_vars_detect_test",
    srcs = ["free_vars_detect_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/247102978)
    ],
    deps = [
        ":free_vars_detect",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
