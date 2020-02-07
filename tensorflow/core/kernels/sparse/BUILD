# Description: Op kernels for sparse matrix operations.

load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_cc_test",
    "tf_kernel_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "sparse_matrix",
    srcs = ["sparse_matrix.cc"],
    hdrs = ["sparse_matrix.h"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
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
        "transpose_op.h",
        "zeros_op.h",
    ],
    gpu_srcs = [
        "zeros_op.h",
        "kernels.h",
        "kernels_gpu.cu.cc",
    ],
    deps = [
        ":sparse_matrix",
        "//third_party/eigen3",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:framework",
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
        "//tensorflow/core/kernels:scatter_nd_op",
        "//tensorflow/core/kernels:slice_op",
        "//tensorflow/core/kernels:transpose_functor",
    ] + if_cuda_or_rocm([
        "//tensorflow/core/kernels:cuda_solvers",
        "//tensorflow/core/kernels:cuda_sparse",
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
        "//third_party/eigen3",
    ],
)
