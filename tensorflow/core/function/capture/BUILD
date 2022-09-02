load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    default_visibility = ["//tensorflow/core/function/capture:__subpackages__"],
)

pytype_strict_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    srcs_version = "PY3",
    deps = [
    ],
)

py_strict_test(
    name = "free_vars_detect_test",
    srcs = ["free_vars_detect_test.py"],
    python_version = "PY3",
    deps = [
        ":free_vars_detect",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
