load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_strict_library(
    name = "dlpack",
    srcs = ["dlpack.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "dlpack_test",
    srcs = ["dlpack_test.py"],
    srcs_version = "PY3",
    deps = [
        ":dlpack",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
