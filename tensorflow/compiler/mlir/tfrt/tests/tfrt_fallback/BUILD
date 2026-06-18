load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests", "mlir_to_bef")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_test")
# copybara:uncomment load("//third_party/tf_runtime_google/cpp_tests:gen_tests.bzl", "tfrt_cc_test_and_strict_benchmark")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
        "//tensorflow/core/runtime_fallback:tf_bef_executor",
        "//tensorflow/core/runtime_fallback/util:fallback_test_util",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:run_lit.sh",
        "@tf_runtime//tools:tfrt_translate",
    ],
)

# copybara:uncomment_begin(TFRT lit issue b/290857552)
# glob_tfrt_lit_tests(
#     data = [":test_utilities"],
#     # Custom driver is unsupported in OSS. Fails if one is provided.
#     # copybara:uncomment driver = "//tensorflow/compiler/mlir:run_lit.sh",
#     exclude = [
#         "compile.benchmark.large.mlir",
#         "batch_function_fallback.mlir",
#         "create_op.mlir",
#         "custom_thread_pool.mlir",
#     ],
#     # copybara:uncomment flaky = ["compile.error.mlir"],
#     size_override = {
#         "compile.benchmark.small.mlir": "medium",
#         "batching_fallback.mlir": "medium",
#     },
#     tags_override = {
#         "async_op_thread.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "batching_fallback.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "compile.benchmark.small.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "convert_tensorhandle_to_fallback_tensor.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "fallback.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "fallback_tensor_conversion_host.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "kernel_fallback_op_handler.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "mnist.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "runtime_fallback_op_handler.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "tf_delegate.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "tf_ops.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "tf_ops_error.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#         "tfrt_forwarding.mlir": ["nomsan"],  # Can't instrument code in precompiled lib (cuDNN)
#     },
#     tfrt_translate = "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
# )
# copybara:uncomment_end

mlir_to_bef(
    name = "batch_function_fallback.mlir",
    tfrt_translate = "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
)

mlir_to_bef(
    name = "create_op.mlir",
    tfrt_translate = "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
)

mlir_to_bef(
    name = "custom_thread_pool.mlir",
    tfrt_translate = "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
)

# copybara:uncomment_begin(internal benchmarking)
# # C++ benchmarks for batch function runtime fallback.
# tfrt_cc_test_and_strict_benchmark(
#     name = "batch_function_fallback_benchmark_test",
#     srcs = ["batch_function_fallback_benchmark_test.cc"],
#     data = ["batch_function_fallback.mlir.bef"],
#     enable_xprof = True,
#     includes = ["third_party/tf_runtime/include"],
#     owners = ["tf-runtime-testing"],
#     tags = [
#         "need_main",
#         "no_gpu",
#     ],
#     deps = [
#         "//base",
#         "//devtools/build/runtime:get_runfiles_dir",
#         "@com_google_absl//absl/log:check",
#         "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
#         "//tensorflow/core/platform:env",
#         "//tensorflow/core/platform:resource_loader",
#         "//tensorflow/core/platform:status",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_op_handler",
#         "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
#         "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
#         "//tensorflow/core/runtime_fallback/util:fallback_test_util",
#         "//tensorflow/core/runtime_fallback/util:tensor_util",
#         "//tensorflow/core/tfrt/utils:fallback_tensor",
#         "@eigen_archive//:eigen3",
#         "@tf_runtime//:bef",
#         "@tf_runtime//:befexecutor",
#         "@tf_runtime//:core_runtime_alwayslink",
#         "@tf_runtime//:hostcontext_alwayslink",
#         "@tf_runtime//:mlirtobef",
#         "@tf_runtime//:support",
#         "@tf_runtime//:tensor",
#         "@tf_runtime//backends/cpu:core_runtime_alwayslink",
#         "@tf_runtime//backends/cpu:test_ops_alwayslink",
#     ],
# )
# copybara:uncomment_end

tf_cc_shared_test(
    name = "kernel_fallback_compat_test",
    srcs = ["kernel_fallback_compat_test.cc"],
    data = [
        "create_op.mlir.bef",
        "custom_thread_pool.mlir.bef",
    ],
    tags = ["no_oss"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:resource_loader",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/runtime_fallback/util:fallback_test_util",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:support",
        "@tf_runtime//:tracing",
    ],
)
