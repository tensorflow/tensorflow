load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:variables",
    ],
)
