# Description:
#    Libraries for helping construct LLVM IR for XLA backends.

load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
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
    name = "alias_analysis",
    srcs = ["alias_analysis.cc"],
    hdrs = ["alias_analysis.h"],
    deps = [
        ":ir_array",
        ":llvm_type_conversion_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:logical_buffer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
    ],
)

xla_cc_test(
    name = "alias_analysis_test",
    srcs = ["alias_analysis_test.cc"],
    deps = [
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:custom_call_target_registry",
        "//xla/service/cpu/tests:cpu_codegen_test",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "llvm_util",
    srcs = ["llvm_util.cc"],
    hdrs = ["llvm_util.h"],
    deps = [
        ":llvm_type_conversion_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_options",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:byte_order",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "llvm_type_conversion_util",
    hdrs = ["llvm_type_conversion_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "llvm_command_line_options",
    hdrs = ["llvm_command_line_options.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ir_array",
    srcs = ["ir_array.cc"],
    hdrs = ["ir_array.h"],
    deps = [
        ":llvm_type_conversion_util",
        ":llvm_util",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "llvm_loop",
    srcs = ["llvm_loop.cc"],
    hdrs = ["llvm_loop.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "loop_emitter",
    srcs = ["loop_emitter.cc"],
    hdrs = ["loop_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_loop",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "fused_ir_emitter",
    srcs = ["fused_ir_emitter.cc"],
    hdrs = ["fused_ir_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        ":tuple_ops",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:fusion_node_indexing_evaluation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:TargetParser",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "dynamic_update_slice_util",
    srcs = ["dynamic_update_slice_util.cc"],
    hdrs = ["dynamic_update_slice_util.h"],
    deps = [
        ":fused_ir_emitter",
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:parallel_loop_emitter",
    ],
)

cc_library(
    name = "sort_util",
    srcs = ["sort_util.cc"],
    hdrs = ["sort_util.h"],
    deps = [
        ":ir_array",
        ":kernel_support_library",
        ":llvm_loop",
        ":llvm_util",
        ":loop_emitter",
        "//xla:shape_util",
        "//xla:util",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/service/gpu:target_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "tuple_ops",
    srcs = ["tuple_ops.cc"],
    hdrs = ["tuple_ops.h"],
    deps = [
        ":ir_array",
        ":llvm_type_conversion_util",
        ":llvm_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "kernel_support_library",
    srcs = ["kernel_support_library.cc"],
    hdrs = ["kernel_support_library.h"],
    deps = [
        ":llvm_loop",
        ":llvm_type_conversion_util",
        ":llvm_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "buffer_assignment_util",
    srcs = ["buffer_assignment_util.cc"],
    hdrs = ["buffer_assignment_util.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "math_ops",
    srcs = ["math_ops.cc"],
    hdrs = ["math_ops.h"],
    deps = [
        ":llvm_util",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "ir_builder_mixin",
    srcs = [],
    hdrs = ["ir_builder_mixin.h"],
    deps = [
        "@llvm-project//llvm:Core",
    ],
)

xla_cc_test(
    name = "ir_array_test",
    srcs = ["ir_array_test.cc"],
    deps = [
        ":ir_array",
        ":llvm_util",
        "//xla:test",
        "//xla/tests:filecheck",
        "//xla/tests:xla_internal_test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
        "@local_tsl//tsl/platform:statusor",
    ],
)
