# Tests of TensorFlow sparse linear algebra kernels using the Python API.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_test(
    name = "conjugate_gradient_test",
    size = "medium",
    srcs = ["conjugate_gradient_test.py"],
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
    name = "csr_sparse_matrix_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_test.py"],
    main = "csr_sparse_matrix_test.py",
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
)

cuda_py_test(
    name = "csr_sparse_matrix_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["csr_sparse_matrix_ops_test.py"],
    main = "csr_sparse_matrix_ops_test.py",
    shard_count = 10,
    tags = [
        "notsan",  # b/149115441
        "no_oss",
    ],
    deps = [
        "//tensorflow/python/ops/linalg/sparse",
        "//tensorflow/python/ops/linalg/sparse:gen_sparse_csr_matrix_ops",
    ],
)

cuda_py_test(
    name = "csr_sparse_matrix_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_grad_test.py"],
    main = "csr_sparse_matrix_grad_test.py",
    shard_count = 50,
    deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
)

cuda_py_test(
    name = "csr_sparse_matrix_dense_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
)

cuda_py_test(
    name = "csr_sparse_matrix_sparse_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_sparse_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_sparse_mat_mul_grad_test.py",
    shard_count = 50,
    deps = [
        "//tensorflow/python/ops/linalg/sparse",
    ],
)
