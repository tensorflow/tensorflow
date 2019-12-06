# Tests of TensorFlow sparse linear algebra kernels using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cuda_py_test(
    name = "conjugate_gradient_test",
    size = "medium",
    srcs = ["conjugate_gradient_test.py"],
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
    name = "csr_sparse_matrix_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
    main = "csr_sparse_matrix_test.py",
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "csr_sparse_matrix_ops_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_ops_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg/sparse",
        "//tensorflow/python/ops/linalg/sparse:gen_sparse_csr_matrix_ops",
    ],
    main = "csr_sparse_matrix_ops_test.py",
    shard_count = 10,
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "csr_sparse_matrix_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_grad_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
    main = "csr_sparse_matrix_grad_test.py",
    shard_count = 50,
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "csr_sparse_matrix_dense_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "csr_sparse_matrix_sparse_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_sparse_mat_mul_grad_test.py"],
    additional_deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
    main = "csr_sparse_matrix_sparse_mat_mul_grad_test.py",
    shard_count = 50,
    tags = ["no_rocm"],
)
