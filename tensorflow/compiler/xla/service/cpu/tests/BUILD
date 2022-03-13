# Description:
#    Tests for LLVM-based CPU backend for XLA.

load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "cpu_codegen_test",
    testonly = True,
    hdrs = ["cpu_codegen_test.h"],
    deps = [
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_dyn_shape_test",
    srcs = ["cpu_dyn_shape_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_fusion_test",
    srcs = ["cpu_fusion_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_instruction_fusion",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "cpu_bytesizeof_test",
    srcs = ["cpu_bytesizeof_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_external_constants_test",
    srcs = ["cpu_external_constants_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:array2d",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "cpu_noalias_test",
    srcs = ["cpu_noalias_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:buffer_assignment",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/compiler/xla/service/llvm_ir:alias_analysis",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Core",
    ],
)

tf_cc_test(
    name = "cpu_intrinsic_test",
    srcs = ["cpu_intrinsic_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

tf_cc_test(
    name = "cpu_eigen_dot_operation_test",
    srcs = ["cpu_eigen_dot_operation_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "cpu_profiling_test",
    srcs = ["cpu_profiling_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

tf_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = ["tree_reduction_rewriter_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/tests:codegen_test_base",
        "//tensorflow/compiler/xla/tests:filecheck",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:llvm_irgen_test_base",
        "//tensorflow/compiler/xla/tests:test_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "cpu_infeed_test",
    srcs = ["cpu_infeed_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:global_data",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/client/lib:arithmetic",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/tests:client_library_test_base",
        "//tensorflow/compiler/xla/tests:literal_test_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_literal_caching_test",
    srcs = ["cpu_literal_caching_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_outfeed_test",
    srcs = ["cpu_outfeed_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_key_value_sort_test",
    srcs = ["cpu_key_value_sort_test.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/service/cpu/tests:cpu_codegen_test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_spmd_compile_test",
    srcs = ["cpu_spmd_compile_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_query",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_topk_test",
    srcs = ["cpu_topk_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:sorting",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/compiler/xla/service/cpu:test_header_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "cpu_vectorization_test",
    srcs = ["cpu_vectorization_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/cpu:cpu_compiler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)
