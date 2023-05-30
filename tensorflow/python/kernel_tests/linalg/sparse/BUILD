# Tests of TensorFlow sparse linear algebra kernels using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "conjugate_gradient_test",
    size = "medium",
    srcs = ["conjugate_gradient_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg/sparse:conjugate_gradient",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_test.py"],
    main = "csr_sparse_matrix_test.py",
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["csr_sparse_matrix_ops_test.py"],
    main = "csr_sparse_matrix_ops_test.py",
    shard_count = 10,
    tags = [
        "no_gpu",  # b/203655060 (cuda 11.5 specific)
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_grad_test.py"],
    main = "csr_sparse_matrix_grad_test.py",
    shard_count = 50,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_dense_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "csr_sparse_matrix_sparse_mat_mul_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_sparse_mat_mul_grad_test.py"],
    main = "csr_sparse_matrix_sparse_mat_mul_grad_test.py",
    shard_count = 50,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradient_checker",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)
