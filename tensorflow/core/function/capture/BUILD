load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    default_visibility = ["//tensorflow/core/function/capture:__subpackages__"],
)

py_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:util",
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
