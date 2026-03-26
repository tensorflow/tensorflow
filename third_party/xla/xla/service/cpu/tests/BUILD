# Description:
#    Tests for LLVM-based CPU backend for XLA.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "tsl_copts")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/mkl:build_defs.bzl", "if_graph_api")
load("//xla/tsl/mkl:graph.bzl", "onednn_cc_test")

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
    name = "cpu_pjrt_codegen_test",
    testonly = True,
    srcs = ["cpu_pjrt_codegen_test.cc"],
    hdrs = ["cpu_pjrt_codegen_test.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/service:llvm_compiler",
        "//xla/service:platform_util",
        "//xla/stream_executor:platform",
        "//xla/tests:codegen_utils",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:pjrt_cpu_client_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:casts",
    ],
)

xla_cc_test(
    name = "cpu_aot_export_test",
    srcs = ["cpu_aot_export_test.cc"],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:cpu_plugin",
        "//xla/service:executable",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

xla_test(
    name = "cpu_fusion_test",
    srcs = ["cpu_fusion_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_instruction_fusion",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "cpu_bytesizeof_test",
    srcs = ["cpu_bytesizeof_test.cc"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "cpu_external_constants_test",
    srcs = ["cpu_external_constants_test.cc"],
    deps = [
        ":cpu_pjrt_codegen_test",
        "//xla:array2d",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_options",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:test",
    ],
)

xla_test(
    name = "cpu_noalias_test",
    srcs = ["cpu_noalias_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:buffer_assignment",
        "//xla/service:compiler",
        "//xla/service:logical_buffer",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/llvm_ir:alias_analysis",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:platform",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

xla_test(
    name = "cpu_ffi_test",
    srcs = ["cpu_ffi_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
    ],
)

xla_cc_test(
    name = "cpu_intrinsic_test",
    srcs = ["cpu_intrinsic_test.cc"],
    # The objective of this test to ensure that LLVM auto vectorization kicks in. This is not
    # necessary with the tiled emitter which explicitly creates vector instructions.
    env = {"XLA_FLAGS": "--xla_backend_extra_options=xla_cpu_disable_tiled_emitter"},
    deps = [
        ":cpu_pjrt_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

xla_cc_test(
    name = "tree_reduction_rewriter_test",
    srcs = ["tree_reduction_rewriter_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/transforms/simplifiers:algebraic_simplifier",
        "//xla/hlo/transforms/simplifiers:tree_reduction_rewriter",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_test(
    name = "cpu_infeed_test",
    srcs = ["cpu_infeed_test.cc"],
    backends = ["cpu"],
    shard_count = 9,
    deps = [
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:arithmetic",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service",
        "//xla/service:hlo_runner_interface",
        "//xla/tests:client_library_test_runner_mixin",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
    ],
)

xla_cc_test(
    name = "cpu_literal_caching_test",
    srcs = ["cpu_literal_caching_test.cc"],
    deps = [
        ":cpu_pjrt_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "cpu_spmd_compile_test",
    srcs = ["cpu_spmd_compile_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:test_header_helper",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "cpu_vectorization_test",
    srcs = ["cpu_vectorization_test.cc"],
    # The objective of this test to ensure that LLVM auto vectorization kicks in. This is not
    # necessary with the tiled emitter which explicitly creates vector instructions.
    env = {"XLA_FLAGS": "--xla_backend_extra_options=xla_cpu_disable_tiled_emitter"},
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:compiler",
        "//xla/service:llvm_compiler",
        "//xla/service:platform_util",
        "//xla/service/cpu:cpu_compiler",
        "//xla/stream_executor:platform",
        "//xla/tests:codegen_utils",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_test(
    name = "cpu_while_test",
    srcs = ["cpu_while_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service/cpu:cpu_compiler",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ARMCodeGen",  # fixdeps: keep
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)

onednn_cc_test(
    name = "onednn_matmul_test",
    srcs = ["onednn_matmul_test.cc"],
    copts = tsl_copts(),
    shard_count = 4,
    tags = [
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
    ],
)

onednn_cc_test(
    name = "onednn_convolution_test",
    srcs = ["onednn_convolution_test.cc"],
    copts = tsl_copts(),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:test",
        "//xla/hlo/testlib:test_helpers",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_contraction_rewriter",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:platform_port",
    ],
)

onednn_cc_test(
    name = "onednn_memory_util_test",
    srcs = ["onednn_memory_util_test.cc"],
    copts = tsl_copts(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_memory_util",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)

xla_cc_test(
    name = "onednn_layer_norm_test",
    srcs = ["onednn_layer_norm_test.cc"],
    copts = tsl_copts(),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

xla_cc_test(
    name = "onednn_softmax_test",
    srcs = ["onednn_softmax_test.cc"],
    copts = tsl_copts(),
    shard_count = if_graph_api(4, 1),
    tags = [
        # TODO: reenable once onednn is supported by the thunk runtime.
        "no_oss",
        "notap",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:pattern_matcher_gmock",
        "//xla/hlo/testlib:test",
        "//xla/service:cpu_plugin",
        "//xla/service:pattern_matcher",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:onednn_config_proto_cc",
        "//xla/service/cpu:onednn_ops_rewriter",
        "//xla/service/cpu:onednn_util",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "onednn_fusion_test",
    srcs = ["onednn_fusion_test.cc"],
    local_defines = if_graph_api(["XLA_ONEDNN_USE_GRAPH_API=1"]),
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla:error_spec",
        "//xla/service:cpu_plugin",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_test(
    name = "cpu_copy_test",
    srcs = ["cpu_copy_test.cc"],
    backends = ["cpu"],
    deps = [
        "//xla:literal",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "hlo_spec_test",
    srcs = ["hlo_spec_test.cc"],
    deps = [
        ":cpu_pjrt_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:cpu_aot_compilation_result",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:AArch64CodeGen",  # fixdeps: keep
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:X86CodeGen",  # fixdeps: keep
    ],
)
