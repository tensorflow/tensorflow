# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "sycl_py_test")

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_test(
    name = "random_shuffle_queue_test",
    size = "small",
    srcs = ["random_shuffle_queue_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
    ],
)

cuda_py_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
    ],
)

cuda_py_test(
    name = "multinomial_op_big_test",
    size = "medium",
    srcs = ["multinomial_op_big_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
    ],
    shard_count = 3,
)

cuda_py_test(
    name = "random_crop_test",
    size = "small",
    srcs = ["random_crop_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:random_ops",
    ],
)

cuda_py_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:random_ops",
    ],
    tags = [
        "manual",
        "no_oss",
    ],
)

cuda_py_test(
    name = "random_gamma_test",
    size = "medium",
    srcs = ["random_gamma_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
    ],
    shard_count = 4,
    tags = ["nozapfhahn"],
)

cuda_py_test(
    name = "random_poisson_test",
    size = "medium",
    srcs = ["random_poisson_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
    ],
)
