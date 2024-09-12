# TF-RISC

load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "risc_abs_op",
    srcs = ["risc_abs_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_add_op",
    srcs = ["risc_add_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_binary_arithmetic_op",
    srcs = ["risc_binary_arithmetic_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_binary_comparison_op",
    srcs = ["risc_binary_comparison_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_bitcast_op",
    srcs = ["risc_bitcast_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_broadcast_op",
    srcs = ["risc_broadcast_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_cast_op",
    srcs = ["risc_cast_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_ceil_op",
    srcs = ["risc_ceil_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_cholesky_op",
    srcs = ["risc_cholesky_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_concat_op",
    srcs = ["risc_concat_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_condition_op",
    srcs = ["risc_condition_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_conv_op",
    srcs = ["risc_conv_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_cos_op",
    srcs = ["risc_cos_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_div_op",
    srcs = ["risc_div_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_dot_op",
    srcs = ["risc_dot_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_exp_op",
    srcs = ["risc_exp_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_fft_op",
    srcs = ["risc_fft_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_floor_op",
    srcs = ["risc_floor_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_gather_op",
    srcs = ["risc_gather_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_imag_op",
    srcs = ["risc_imag_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_is_finite_op",
    srcs = ["risc_is_finite_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_log_op",
    srcs = ["risc_log_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_logical_and_op",
    srcs = ["risc_logical_and_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_logical_not_op",
    srcs = ["risc_logical_not_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_logical_or_op",
    srcs = ["risc_logical_or_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_max_op",
    srcs = ["risc_max_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_min_op",
    srcs = ["risc_min_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_mul_op",
    srcs = ["risc_mul_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_neg_op",
    srcs = ["risc_neg_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_pad_op",
    srcs = ["risc_pad_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_pool_op",
    srcs = ["risc_pool_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_pow_op",
    srcs = ["risc_pow_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_random_uniform_op",
    srcs = ["risc_random_uniform_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_real_op",
    srcs = ["risc_real_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_reduce_op",
    srcs = ["risc_reduce_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_rem_op",
    srcs = ["risc_rem_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_reshape_op",
    srcs = ["risc_reshape_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_reverse_op",
    srcs = ["risc_reverse_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_scatter_op",
    srcs = ["risc_scatter_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_shape_op",
    srcs = ["risc_shape_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "risc_sign_op",
    srcs = ["risc_sign_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_slice_op",
    srcs = ["risc_slice_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_sort_op",
    srcs = ["risc_sort_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_squeeze_op",
    srcs = ["risc_squeeze_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_sub_op",
    srcs = ["risc_sub_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "risc_transpose_op",
    srcs = ["risc_transpose_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_triangular_solve_op",
    srcs = ["risc_triangular_solve_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_unary_op",
    srcs = ["risc_unary_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "risc_while_op",
    srcs = ["risc_while_op.cc"],
    deps = ["//tensorflow/core:framework"],
)

tf_kernel_library(
    name = "experimental",
    deps = [
        ":risc_abs_op",
        ":risc_add_op",
        ":risc_binary_arithmetic_op",
        ":risc_binary_comparison_op",
        ":risc_bitcast_op",
        ":risc_broadcast_op",
        ":risc_cast_op",
        ":risc_ceil_op",
        ":risc_cholesky_op",
        ":risc_condition_op",
        ":risc_conv_op",
        ":risc_cos_op",
        ":risc_div_op",
        ":risc_dot_op",
        ":risc_exp_op",
        ":risc_fft_op",
        ":risc_floor_op",
        ":risc_gather_op",
        ":risc_imag_op",
        ":risc_is_finite_op",
        ":risc_log_op",
        ":risc_logical_and_op",
        ":risc_logical_not_op",
        ":risc_logical_or_op",
        ":risc_max_op",
        ":risc_min_op",
        ":risc_mul_op",
        ":risc_neg_op",
        ":risc_pad_op",
        ":risc_pool_op",
        ":risc_pow_op",
        ":risc_random_uniform_op",
        ":risc_real_op",
        ":risc_reduce_op",
        ":risc_rem_op",
        ":risc_reshape_op",
        ":risc_reverse_op",
        ":risc_scatter_op",
        ":risc_shape_op",
        ":risc_sign_op",
        ":risc_slice_op",
        ":risc_sort_op",
        ":risc_squeeze_op",
        ":risc_sub_op",
        ":risc_transpose_op",
        ":risc_triangular_solve_op",
        ":risc_unary_op",
        ":risc_while_op",
    ],
)
