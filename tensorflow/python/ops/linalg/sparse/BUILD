load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")

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
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    visibility = ["//visibility:private"],
    deps = ["//tensorflow/core:sparse_csr_matrix_ops_op_lib"],
)

py_strict_library(
    name = "sparse_py",
    srcs = ["sparse.py"],
    deps = [
        ":conjugate_gradient",
        ":sparse_csr_matrix_grad",
        ":sparse_csr_matrix_ops",
    ],
)

py_strict_library(
    name = "sparse_csr_matrix_grad",
    srcs = ["sparse_csr_matrix_grad.py"],
    deps = [
        ":sparse_csr_matrix_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.py"],
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/linalg:linalg_impl",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "sparse_csr_matrix_ops",
    srcs = ["sparse_csr_matrix_ops.py"],
    deps = [
        ":gen_sparse_csr_matrix_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:cpp_shape_inference_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
    ],
)
