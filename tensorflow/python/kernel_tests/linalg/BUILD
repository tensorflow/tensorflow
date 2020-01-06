# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cuda_py_test(
    name = "linear_operator_test",
    size = "small",
    srcs = ["linear_operator_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_addition_test",
    size = "small",
    srcs = ["linear_operator_addition_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_adjoint_test",
    size = "medium",
    srcs = ["linear_operator_adjoint_test.py"],
    shard_count = 5,
    tags = [
        "noasan",  # times out, b/63678675
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_algebra_test",
    size = "small",
    srcs = ["linear_operator_algebra_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_block_diag_test",
    size = "medium",
    srcs = ["linear_operator_block_diag_test.py"],
    shard_count = 6,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_composition_test",
    size = "medium",
    srcs = ["linear_operator_composition_test.py"],
    shard_count = 5,
    tags = [
        "noasan",  # times out, b/63678675
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_circulant_test",
    size = "medium",
    srcs = ["linear_operator_circulant_test.py"],
    shard_count = 10,
    tags = [
        "no_rocm",  # calls BLAS ops for complex types
        "noasan",  # times out, b/63678675
        "optonly",  # times out, b/79171797
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/signal",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_diag_test",
    size = "medium",
    srcs = ["linear_operator_diag_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_householder_test",
    size = "medium",
    srcs = ["linear_operator_householder_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_identity_test",
    size = "medium",
    srcs = ["linear_operator_identity_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_inversion_test",
    size = "medium",
    srcs = ["linear_operator_inversion_test.py"],
    shard_count = 5,
    tags = [
        "noasan",  # times out, b/63678675
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_full_matrix_test",
    size = "medium",
    srcs = ["linear_operator_full_matrix_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_kronecker_test",
    size = "medium",
    srcs = ["linear_operator_kronecker_test.py"],
    shard_count = 10,
    tags = [
        "noasan",
        "optonly",
    ],
    xla_enable_strict_auto_jit = False,  # Fails in XLA b/143610154
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_lower_triangular_test",
    size = "medium",
    srcs = ["linear_operator_lower_triangular_test.py"],
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_low_rank_update_test",
    size = "medium",
    srcs = ["linear_operator_low_rank_update_test.py"],
    shard_count = 10,
    tags = [
        "noasan",  # times out
        "optonly",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_permutation_test",
    size = "medium",
    srcs = ["linear_operator_permutation_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_util_test",
    size = "medium",
    srcs = ["linear_operator_util_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_toeplitz_test",
    size = "medium",
    srcs = ["linear_operator_toeplitz_test.py"],
    shard_count = 5,
    tags = [
        "no_cuda_on_cpu_tap",  # flaky, b/135701551
        "no_gpu",  # flaky, b/135701551
        "noasan",  # times out, b/63678675
        "optonly",  # times out, b/79171797
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/signal",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "linear_operator_tridiag_test",
    size = "medium",
    srcs = ["linear_operator_tridiag_test.py"],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)

cuda_py_test(
    name = "linear_operator_zeros_test",
    size = "medium",
    srcs = ["linear_operator_zeros_test.py"],
    shard_count = 5,
    tags = ["optonly"],  # Test is flaky without optimization.
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/ops/linalg",
    ],
)
