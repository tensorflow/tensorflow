# Tests of TensorFlow kernels written using the Python API.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cuda_py_test(
    name = "linear_operator_test",
    size = "small",
    srcs = ["linear_operator_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "linear_operator_block_diag_test",
    size = "medium",
    srcs = ["linear_operator_block_diag_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 6,
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_composition_test",
    size = "medium",
    srcs = ["linear_operator_composition_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = [
        "noasan",  # times out, b/63678675
        "optonly",  # times out
    ],
)

cuda_py_test(
    name = "linear_operator_circulant_test",
    size = "medium",
    srcs = ["linear_operator_circulant_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:spectral_ops_test_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = [
        "noasan",  # times out, b/63678675
        "optonly",  # times out, b/79171797
    ],
)

cuda_py_test(
    name = "linear_operator_diag_test",
    size = "medium",
    srcs = ["linear_operator_diag_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_identity_test",
    size = "medium",
    srcs = ["linear_operator_identity_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_full_matrix_test",
    size = "medium",
    srcs = ["linear_operator_full_matrix_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_kronecker_test",
    size = "medium",
    srcs = ["linear_operator_kronecker_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 8,
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_lower_triangular_test",
    size = "medium",
    srcs = ["linear_operator_lower_triangular_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "noasan",
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_low_rank_update_test",
    size = "large",
    srcs = ["linear_operator_low_rank_update_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = [
        "noasan",  # times out
        "optonly",
    ],
)

cuda_py_test(
    name = "linear_operator_util_test",
    size = "medium",
    srcs = ["linear_operator_util_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
)
