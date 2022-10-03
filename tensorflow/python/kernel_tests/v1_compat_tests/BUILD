load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
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

cuda_py_test(
    name = "scatter_nd_ops_test",
    size = "small",
    srcs = ["scatter_nd_ops_test.py"],
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "session_ops_test",
    size = "small",
    srcs = ["session_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session_ops",
    ],
)

cuda_py_test(
    name = "array_ops_test",
    size = "small",
    srcs = ["array_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
    ],
)

cuda_py_test(
    name = "stack_op_test",
    size = "small",
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

cuda_py_test(
    name = "dense_update_ops_test",
    size = "small",
    srcs = ["dense_update_ops_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)
