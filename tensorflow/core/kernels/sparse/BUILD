# Description: Op kernels for sparse matrix operations.

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "sparse_matrix",
    srcs = ["sparse_matrix.cc"],
    hdrs = ["sparse_matrix.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "kernels",
    srcs = [
        "add_op.cc",
        "conj_op.cc",
        "csr_sparse_matrix_to_dense_op.cc",
        "csr_sparse_matrix_to_sparse_tensor_op.cc",
        "dense_to_csr_sparse_matrix_op.cc",
        "kernels.cc",
        "mat_mul_op.cc",
        "mul_op.cc",
        "nnz_op.cc",
        "softmax_op.cc",
        "sparse_cholesky_op.cc",
        "sparse_mat_mul_op.cc",
        "sparse_matrix_components_op.cc",
        "sparse_ordering_amd_op.cc",
        "sparse_tensor_to_csr_sparse_matrix_op.cc",
        "transpose_op.cc",
        "zeros_op.cc",
    ],
    hdrs = [
        "kernels.h",
        "mat_mul_op.h",
        "transpose_op.h",
        "zeros_op.h",
    ],
    gpu_deps = [
        "@local_xla//xla/stream_executor/cuda:cuda_activation",
        "//tensorflow/core/kernels:gpu_device_array",
    ],
    gpu_srcs = [
        "zeros_op.h",
        "kernels.h",
        "kernels_gpu.cu.cc",
    ],
    deps = [
        ":sparse_matrix",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core:no_op_op_lib",
        "//tensorflow/core:sendrecv_ops_op_lib",
        "//tensorflow/core:sparse_csr_matrix_ops_op_lib",
        "//tensorflow/core:state_ops_op_lib",
        "//tensorflow/core/kernels:concat_lib",
        "//tensorflow/core/kernels:constant_op",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:dense_update_functor",
        "//tensorflow/core/kernels:fill_functor",
        "//tensorflow/core/kernels:gather_nd_op",
        "//tensorflow/core/kernels:gpu_prim_hdrs",
        "//tensorflow/core/kernels:scatter_nd_op",
        "//tensorflow/core/kernels:slice_op",
        "//tensorflow/core/kernels:sparse_utils",
        "//tensorflow/core/kernels:transpose_functor",
        "@eigen_archive//:eigen3",
    ] + if_cuda_or_rocm([
        "//tensorflow/core/util:cuda_solvers",
        "//tensorflow/core/util:cuda_sparse",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "kernels_test",
    size = "small",
    srcs = [
        "kernels_test.cc",
    ],
    deps = [
        ":kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)
