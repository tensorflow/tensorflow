# Description: Sparse CSR support for TensorFlow.
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_gen_op_wrapper_py(
    name = "gen_sparse_csr_matrix_ops",
    out = "gen_sparse_csr_matrix_ops.py",
    api_def_srcs = ["//tensorflow/core/api_def:base_api_def"],
    visibility = ["//tensorflow/python/kernel_tests/linalg/sparse:__pkg__"],
    deps = ["//tensorflow/core:sparse_csr_matrix_ops_op_lib"],
)

py_library(
    name = "sparse",
    srcs = [
        "__init__.py",
        "conjugate_gradient.py",
        "sparse.py",
        "sparse_csr_matrix_grad.py",
        "sparse_csr_matrix_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":gen_sparse_csr_matrix_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/linalg:linalg_impl",
        "//third_party/py/numpy",
    ],
)
