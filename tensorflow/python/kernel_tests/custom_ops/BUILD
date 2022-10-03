# Tests of TensorFlow custom ops written

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(licenses = ["notice"])

# Custom op tests
tf_custom_op_library(
    name = "ackermann_op.so",
    srcs = ["ackermann_op.cc"],
)

tf_py_test(
    name = "ackermann_test",
    size = "small",
    srcs = ["ackermann_test.py"],
    data = [":ackermann_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_custom_op_library(
    name = "duplicate_op.so",
    srcs = ["duplicate_op.cc"],
)

tf_py_test(
    name = "duplicate_op_test",
    size = "small",
    srcs = ["duplicate_op_test.py"],
    data = [":duplicate_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_custom_op_library(
    name = "invalid_op.so",
    srcs = ["invalid_op.cc"],
)

tf_py_test(
    name = "invalid_op_test",
    size = "small",
    srcs = ["invalid_op_test.py"],
    data = [":invalid_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)
