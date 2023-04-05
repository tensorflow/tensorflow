# Description:
#   XLA client libraries.

load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
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
    name = "global_data",
    srcs = ["global_data.cc"],
    hdrs = ["global_data.h"],
    deps = [
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/tsl/lib/math:math_util",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "padding_test",
    srcs = ["padding_test.cc"],
    deps = [
        ":padding",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        ":global_data",
        ":xla_computation",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:service_interface",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "executable_build_options",
    srcs = ["executable_build_options.cc"],
    hdrs = ["executable_build_options.h"],
    deps = [
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:execution_options_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/pjrt:compile_options_proto_cc",
        "//tensorflow/compiler/xla/service:compilation_environments",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "local_client",
    srcs = ["local_client.cc"],
    hdrs = ["local_client.h"],
    deps = [
        ":client",
        ":executable_build_options",
        ":xla_computation",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/service:source_map_util",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:TargetParser",
    ],
)

cc_library(
    name = "compile_only_client",
    srcs = ["compile_only_client.cc"],
    hdrs = ["compile_only_client.h"],
    deps = [
        ":client",
        ":xla_computation",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:compile_only_service",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/stream_executor",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

# This target is used to instantiate the XLA service in-process and create
# a client for it.
cc_library(
    name = "client_library",
    srcs = ["client_library.cc"],
    hdrs = ["client_library.h"],
    deps = [
        ":compile_only_client",
        ":local_client",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:compile_only_service",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/tsl/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "sharding_builder",
    srcs = ["sharding_builder.cc"],
    hdrs = ["sharding_builder.h"],
    deps = [
        "//tensorflow/compiler/xla:array",
        "//tensorflow/compiler/xla:shape_tree",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
    ],
)

cc_library(
    name = "xla_computation",
    srcs = ["xla_computation.cc"],
    hdrs = ["xla_computation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "value_inference",
    srcs = ["value_inference.cc"],
    hdrs = ["value_inference.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_builder",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/evaluator:hlo_evaluator",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xla_builder",
    srcs = ["xla_builder.cc"],
    hdrs = ["xla_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":padding",
        ":sharding_builder",
        ":xla_computation",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:sharding_op_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/tsl/lib/core:bitmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:stacktrace",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "xla_builder_test",
    srcs = ["xla_builder_test.cc"],
    deps = [
        ":sharding_builder",
        ":value_inference",
        ":xla_builder",
        ":xla_computation",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:test_helpers",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/hlo/utils:hlo_matchers",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "//tensorflow/tsl/platform:test",
    ],
)
