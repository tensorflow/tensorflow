load(
    "//tensorflow:tensorflow.bzl",
    "if_oss",
    "tf_cc_binary",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

cc_library(
    name = "benchmark",
    testonly = 1,
    srcs = ["benchmark.cc"],
    hdrs = ["benchmark.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_pipeline",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/compiler/xla/runtime:types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:logging",
        "//third_party/eigen3",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/jitrt:async_task_runner",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
        "@tf_runtime//backends/jitrt:results",
    ],
)

cc_library(
    name = "benchmark_mlir_function",
    testonly = 1,
    srcs = ["benchmark_mlir_function.cc"],
    hdrs = ["benchmark_mlir_function.h"],
    deps = [
        ":benchmark",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:host_context_util",
        "//tensorflow/compiler/mlir/tfrt:runtime_fallback_executor",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_pipeline",
        "//tensorflow/compiler/mlir/tfrt:tf_to_tfrt",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
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
        "//tensorflow/compiler/mlir/tfrt:host_context_util",
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
    deps = [
        ":benchmark",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tfrt:host_context_util",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_binary(
    name = "transpose_op_benchmark",
    testonly = 1,
    srcs = ["transpose_op_benchmark.cc"],
    deps = [
        ":benchmark_mlir_function",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "reduction_benchmark",
    testonly = 1,
    srcs = ["reduction_benchmark.cc"],
    hdrs = ["reduction_benchmark.h"],
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
    ],
)

tf_cc_binary(
    name = "softmax_op_benchmark",
    testonly = 1,
    srcs = ["softmax_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_binary(
    name = "sum_full_op_benchmark",
    testonly = 1,
    srcs = ["sum_full_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        ":reduction_benchmark",
    ],
)

tf_cc_binary(
    name = "sum_transposed_op_benchmark",
    testonly = 1,
    srcs = ["sum_transposed_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        ":reduction_benchmark",
    ],
)

tf_cc_binary(
    name = "sum_col_op_benchmark",
    testonly = 1,
    srcs = ["sum_col_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        ":reduction_benchmark",
    ],
)

tf_cc_binary(
    name = "sum_row_op_benchmark",
    testonly = 1,
    srcs = ["sum_row_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        ":reduction_benchmark",
    ],
)

tf_cc_binary(
    name = "mean_row_op_benchmark",
    testonly = 1,
    srcs = ["mean_row_op_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        ":reduction_benchmark",
    ],
)

tf_cc_binary(
    name = "fused_reduction_benchmark",
    testonly = 1,
    srcs = ["fused_reduction_benchmark.cc"],
    # Args() not supported. Enable when we got rid of tf benchmark and use the
    # standard gunit benchmark.
    tags = if_oss([
        "no_oss",
        "manual",
    ]),
    deps = [
        ":benchmark",
        ":benchmark_mlir_function",
        "@llvm-project//llvm:Support",
    ],
)
