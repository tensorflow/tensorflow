package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "module",
    srcs = ["module.py"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/tracking",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    additional_deps = [
        ":module",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python:variables",
    ],
)
