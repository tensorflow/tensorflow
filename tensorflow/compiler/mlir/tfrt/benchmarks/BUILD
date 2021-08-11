load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
    "tf_cc_binary",
    "tf_cc_test",
)

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "benchmark",
    testonly = 1,
    srcs = ["benchmark.cc"],
    hdrs = ["benchmark.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_passes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
        "//third_party/eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)

cc_library(
    name = "benchmark_mlir_function",
    testonly = 1,
    srcs = ["benchmark_mlir_function.cc"],
    hdrs = ["benchmark_mlir_function.h"],
    deps = [
        ":benchmark",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)

tf_cc_binary(
    name = "compute_function_benchmark",
    testonly = 1,
    srcs = ["compute_function_benchmark.cc"],
    deps = [":benchmark_mlir_function"],
)

tf_cc_test(
    name = "cwise_op_exp_benchmark",
    testonly = 1,
    srcs = ["cwise_op_exp_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_expm1_benchmark",
    testonly = 1,
    srcs = ["cwise_op_expm1_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_binary(
    name = "cwise_op_fusion_benchmark",
    testonly = 1,
    srcs = ["cwise_op_fusion_benchmark.cc"],
    deps = [":benchmark_mlir_function"],
)

tf_cc_test(
    name = "cwise_op_log1p_benchmark",
    testonly = 1,
    srcs = ["cwise_op_log1p_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_log2_benchmark",
    testonly = 1,
    srcs = ["cwise_op_log2_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_log_benchmark",
    testonly = 1,
    srcs = ["cwise_op_log_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_rsqrt_benchmark",
    testonly = 1,
    srcs = ["cwise_op_rsqrt_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_sigmoid_benchmark",
    testonly = 1,
    srcs = ["cwise_op_sigmoid_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

tf_cc_test(
    name = "cwise_op_tanh_benchmark",
    testonly = 1,
    srcs = ["cwise_op_tanh_benchmark.cc"],
    deps = [":cwise_op_unary_benchmark"],
)

cc_library(
    name = "cwise_op_unary_benchmark",
    testonly = 1,
    hdrs = ["cwise_op_unary_benchmark.h"],
    deps = [
        ":benchmark",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_binary(
    name = "matmul_op_benchmark",
    testonly = 1,
    srcs = [
        "matmul_op_benchmark.cc",
        "matmul_op_benchmark.h",
    ],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [":benchmark"],
)

tf_cc_binary(
    name = "transpose_op_benchmark",
    testonly = 1,
    srcs = ["transpose_op_benchmark.cc"],
    deps = [":benchmark_mlir_function"],
)
