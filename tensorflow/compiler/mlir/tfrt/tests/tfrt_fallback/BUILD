load("@tf_runtime//tools:mlir_to_bef.bzl", "glob_tfrt_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    srcs = [
        "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_translate",
        "//tensorflow/core/runtime_fallback:tf_bef_executor",
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
