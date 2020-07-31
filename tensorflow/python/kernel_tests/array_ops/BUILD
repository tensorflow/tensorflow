# Tests of TensorFlow array ops kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cuda_py_test(
    name = "batch_gather_op_test",
    srcs = ["batch_gather_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "unstack_op_test",
    size = "small",
    srcs = ["unstack_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "slice_op_test",
    size = "medium",
    srcs = ["slice_op_test.py"],
    tags = ["no_windows"],  # b/126916429
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
    ],
)
