# QuantizationDebugger for TFLite accuracy tooling.
load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "debugger",
    srcs = ["debugger.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:interpreter",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "debugger_test",
    srcs = [
        "debugger_test.py",
    ],
    strict_deps = True,
    deps = [
        ":debugger",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:lite",
        "//tensorflow/lite/python/metrics",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/trackable:autotrackable",
    ],
)
