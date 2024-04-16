# Description:
#    Tests for LLVM-based CPU backend for XLA.

load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "tsl_copts")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
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
        "//xla/service:cpu_plugin",
        "//xla/tests:llvm_irgen_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_aot_export_test",
    srcs = ["cpu_aot_export_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/stream_executor",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_dyn_shape_test",
    srcs = ["cpu_dyn_shape_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_fusion_test",
    srcs = ["cpu_fusion_test.cc"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:cpu_instruction_fusion",
        "//xla/tests:hlo_test_base",
        "//xla/tests:literal_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_bytesizeof_test",
    srcs = ["cpu_bytesizeof_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_external_constants_test",
    srcs = ["cpu_external_constants_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:array2d",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@local_tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "cpu_noalias_test",
    srcs = ["cpu_noalias_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_ordering",
        "//xla/service:logical_buffer",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:filecheck",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_intrinsic_test",
    srcs = ["cpu_intrinsic_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_eigen_dot_operation_test",
    srcs = ["cpu_eigen_dot_operation_test.cc"],
    copts = tsl_copts(),
    tags = ["no_mac_arm64"],
    deps = [
        ":cpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/tests:test_utils",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_profiling_test",
    srcs = ["cpu_profiling_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = ["tree_reduction_rewriter_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/service/cpu:cpu_compiler",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_infeed_test",
    srcs = ["cpu_infeed_test.cc"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test_helpers",
        "//xla:xla_data_proto_cc",
        "//xla/client:global_data",
        "//xla/client:local_client",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:arithmetic",
        "//xla/service:cpu_plugin",
        "//xla/tests:client_library_test_base",
        "//xla/tests:literal_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_literal_caching_test",
    srcs = ["cpu_literal_caching_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_outfeed_test",
    srcs = ["cpu_outfeed_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_key_value_sort_test",
    srcs = ["cpu_key_value_sort_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_spmd_compile_test",
    srcs = ["cpu_spmd_compile_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_topk_test",
    srcs = ["cpu_topk_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/client/lib:sorting",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_vectorization_test",
    srcs = ["cpu_vectorization_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_while_test",
    srcs = ["cpu_while_test.cc"],
    deps = [
        ":cpu_codegen_test",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test_main",
    ],
)
