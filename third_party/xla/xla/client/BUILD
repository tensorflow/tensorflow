# Description:
#   XLA client libraries.

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
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
    name = "padding",
    hdrs = ["padding.h"],
    deprecation = "This library is deprecated and will be removed in February 2025. Use //third_party/tensorflow/compiler/xla/hlo/builder:padding instead.",
    deps = [
        "//xla/hlo/builder:padding",
    ],
)

cc_library(
    name = "client",
    srcs = ["client.cc"],
    hdrs = ["client.h"],
    deps = [
        "//xla:execution_options_util",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/service",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "executable_build_options",
    srcs = ["executable_build_options.cc"],
    hdrs = ["executable_build_options.h"],
    deps = [
        "//xla:debug_options_flags",
        "//xla:execution_options_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:compilation_environments",
        "//xla/service:computation_placer",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "executable_build_options_test",
    srcs = ["executable_build_options_test.cc"],
    deps = [
        ":executable_build_options",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/service:computation_placer",
        "//xla/service:test_compilation_environment_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "local_client",
    srcs = ["local_client.cc"],
    hdrs = ["local_client.h"],
    deps = [
        ":client",
        ":executable_build_options",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:computation_layout",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:local_service",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:source_map_util",
        "//xla/service:stream_pool",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "compile_only_client",
    srcs = ["compile_only_client.cc"],
    hdrs = ["compile_only_client.h"],
    deps = [
        ":client",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/service:compile_only_service",
        "//xla/service:compiler",
        "//xla/service:hlo_module_config",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        "//xla:types",
        "//xla/service",
        "//xla/service:compile_only_service",
        "//xla/service:local_service",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "sharding_builder",
    hdrs = ["sharding_builder.h"],
    deprecation = "This library is deprecated and will be removed in February 2025. Use //third_party/tensorflow/compiler/xla/hlo/builder:sharding_builder instead.",
    deps = [
        "//xla/hlo/builder:sharding_builder",
    ],
)

cc_library(
    name = "xla_computation",
    hdrs = ["xla_computation.h"],
    deprecation = "This library is deprecated and will be removed in February 2025. Use //third_party/tensorflow/compiler/xla/hlo/builder:xla_computation instead.",
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "value_inference",
    hdrs = ["value_inference.h"],
    deprecation = "This library is deprecated and will be removed in February 2025. Use //third_party/tensorflow/compiler/xla/hlo/builder:value_inference instead.",
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/builder:value_inference",
    ],
)

cc_library(
    name = "xla_builder",
    hdrs = ["xla_builder.h"],
    deprecation = "This library is deprecated and will be removed in February 2025. Use //third_party/tensorflow/compiler/xla/hlo/builder:xla_builder instead.",
    visibility = ["//visibility:public"],
    deps = [
        "//xla/hlo/builder:xla_builder",
    ],
)
