# Tests of TensorFlow sparse linear algebra kernels using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/linalg",
        "//tensorflow/python/ops/linalg/sparse:conjugate_gradient",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
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
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
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
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
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
    shard_count = 3,
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
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
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "csr_sparse_matrix_dense_mat_mul_onednn_grad_test",
    size = "medium",
    srcs = ["csr_sparse_matrix_dense_mat_mul_grad_test.py"],
    env = {
        "TF_ENABLE_ONEDNN_SPMM": "1",
    },
    main = "csr_sparse_matrix_dense_mat_mul_grad_test.py",
    shard_count = 50,
    tags = [
        "no_cuda_asan",  # TODO(b/190824595)
    ],
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
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
    shard_count = 10,
    deps = [
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_grad",
        "//tensorflow/python/ops/linalg/sparse:sparse_csr_matrix_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
    ],
)
