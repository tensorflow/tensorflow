load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "module",
    srcs = ["module.py"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/tracking",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":module",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "@absl_py//absl/testing:parameterized",
    ],
)
