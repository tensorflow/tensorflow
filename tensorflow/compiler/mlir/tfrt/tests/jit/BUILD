load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(JitRt deprecated)
# glob_lit_tests(
#     name = "all_tests",
#     data = [":test_utilities"],
#     driver = "//tensorflow/compiler/mlir:run_lit.sh",
#     exclude = ["tf_jitrt_stream_analysis.mlir"],  # TFRT JitRe deprecated
#     features = if_oss(["--path=org_tensorflow/tensorflow/compiler/mlir/tfrt"]),
#     test_file_exts = ["mlir"],
# )
#
# # Bundle together all of the test utilities that are used by tests.
# filegroup(
#     name = "test_utilities",
#     testonly = True,
#     data = [
#         "@llvm-project//llvm:FileCheck",
#         "@llvm-project//llvm:not",
#         "@llvm-project//mlir:run_lit.sh",
#         "//tensorflow/compiler/mlir/tfrt:tf-tfrt-opt",
#     ],
# )
#
# tfrt_cc_test(
#     name = "tf_jitrt_benchmark_test",
#     srcs = ["tf_jitrt_benchmark_test.cc"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:BufferizationTransforms",
#         "//tensorflow/compiler/jit:flags",
#         "//tensorflow/compiler/mlir/tensorflow",
#         "//tensorflow/compiler/mlir/tfrt:tf_jitrt_pipeline",
#         "//tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
#         "//tensorflow/compiler/xla/runtime:executable",
#         "//tensorflow/compiler/xla/runtime:jit_executable",
#         "//tensorflow/core:test",
#         "//tensorflow/core:test_main",
#         "@tf_runtime//:dtype",
#         "@tf_runtime//backends/jitrt:jitrt_compiler",
#     ],
# )
# copybara:uncomment_end
