# Tests of TensorFlow array ops kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
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
    tags = [
        "no_cuda_asan",  # b/173806679
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
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
    name = "gather_op_test",
    size = "medium",
    srcs = ["gather_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "scatter_nd_ops_test",
    size = "medium",
    srcs = ["scatter_nd_ops_test.py"],
    tags = ["noasan"],  # http://b/32635055
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "stack_op_test",
    size = "medium",
    srcs = ["stack_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)
