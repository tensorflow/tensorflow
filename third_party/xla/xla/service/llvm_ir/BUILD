# Description:
#    Libraries for helping construct LLVM IR for XLA backends.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

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
        "//xla:shape_util",
        "//xla:types",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_value",
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
        "//xla:xla_proto_cc",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/service/cpu/tests:cpu_codegen_test_main",
        "//xla/tests:hlo_test_base",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "llvm_util",
    srcs = ["llvm_util.cc"],
    hdrs = ["llvm_util.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    deps = [
        ":llvm_type_conversion_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:dump",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:cpu_options",
        "//xla/tsl/platform:byte_order",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
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
    srcs = ["llvm_command_line_options.cc"],
    hdrs = ["llvm_command_line_options.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
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
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
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
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
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
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "fused_ir_emitter",
    srcs = ["fused_ir_emitter.cc"],
    hdrs = ["fused_ir_emitter.h"],
    deps = [
        ":ir_array",
        ":llvm_util",
        ":loop_emitter",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:elemental_ir_emitter",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
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
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:elemental_ir_emitter",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:ir_headers",
    ],
)

cc_library(
    name = "sort_util",
    srcs = ["sort_util.cc"],
    hdrs = ["sort_util.h"],
    deps = [
        ":ir_array",
        ":kernel_support_library",
        ":llvm_util",
        ":loop_emitter",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/service/gpu:launch_dimensions",
        "//xla/service/gpu:parallel_loop_emitter",
        "//xla/service/gpu:target_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
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
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
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
        "//xla/service:hlo_module_config",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Core",
    ],
)

cc_library(
    name = "buffer_assignment_util",
    srcs = ["buffer_assignment_util.cc"],
    hdrs = ["buffer_assignment_util.h"],
    deps = [
        "//xla:literal",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "@com_google_absl//absl/log:check",
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
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:filecheck",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:ir_headers",
    ],
)
