# Build multiplex_1 custom ops examples, which is similar to np.where

load("//tensorflow:tensorflow.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_1_kernel.so",
    srcs = [
        "multiplex_1_kernel.cc",
        "multiplex_1_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_1_op",
    srcs = ["multiplex_1_op.py"],
    data = [":multiplex_1_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "multiplex_1_test",
    size = "small",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
        "no_pip",
    ],
    deps = [
        ":multiplex_1_op",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
