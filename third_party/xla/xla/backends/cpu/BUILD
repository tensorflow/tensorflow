load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/backends/cpu:package_groups.bzl", "xla_cpu_backend_access")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("//xla/tsl/mkl:graph.bzl", "onednn_graph_cc_library")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

xla_cpu_backend_access()

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "alignment.h",
        "buffer_allocation_info.h",
    ],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "alignment",
    hdrs = ["alignment.h"],
    visibility = internal_visibility([":friends"]),
    deps = ["@eigen_archive//:eigen3"],
)

cc_library(
    name = "buffer_allocation_info",
    hdrs = ["buffer_allocation_info.h"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "buffer_allocation_info_util",
    srcs = ["buffer_allocation_info_util.cc"],
    hdrs = ["buffer_allocation_info_util.h"],
    visibility = internal_visibility([":friends"]),
    deps = [
        ":buffer_allocation_info",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/ffi",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
    ],
)

onednn_graph_cc_library(
    name = "onednn_emitter",
    srcs = ["onednn_emitter.cc"],
    hdrs = ["onednn_emitter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":onednn_fusion",
        ":onednn_support",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_dims",
        "//xla/backends/cpu/runtime/onednn:onednn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/mkl:onednn",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

onednn_graph_cc_library(
    name = "onednn_fusion",
    hdrs = ["onednn_fusion.h"],
    compatible_with = get_compatible_with_portable(),
    deps = ["//xla/tsl/mkl:onednn"],
)

onednn_graph_cc_library(
    name = "onednn_support",
    srcs = ["onednn_support.cc"],
    hdrs = ["onednn_support.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:target_machine_features",
        "//xla/backends/cpu/runtime:dot_dims",
        "//xla/hlo/ir:hlo",
        "//xla/service/cpu:onednn_util",
        "//xla/tsl/mkl:onednn",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:platform_port",
    ],
)

# TODO: b/467367981, this is deprecated and should be removed.
tf_proto_library(
    name = "xnn_fusion_options_proto",
    srcs = ["xnn_fusion_options.proto"],
)

cc_library(
    name = "ynn_emitter",
    srcs = ["ynn_emitter.cc"],
    hdrs = ["ynn_emitter.h"],
    deps = [
        ":ynn_support",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_dims",
        "//xla/backends/cpu/runtime/ynnpack:ynn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_address",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK//ynnpack",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

tf_proto_library(
    name = "ynn_fusion_options_proto",
    srcs = ["ynn_fusion_options.proto"],
)

cc_library(
    name = "ynn_support",
    srcs = ["ynn_support.cc"],
    hdrs = ["ynn_support.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/runtime:dot_dims",
        "//xla/backends/cpu/runtime/ynnpack:ynn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/tsl/platform:statusor",
        "@XNNPACK//ynnpack",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "target_machine_options",
    srcs = ["target_machine_options.cc"],
    hdrs = ["target_machine_options.h"],
    deps = [
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/backends/cpu/codegen:cpu_features",
        "//xla/service/cpu:executable_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TargetParser",
    ],
)

xla_cc_test(
    name = "target_machine_options_test",
    srcs = ["target_machine_options_test.cc"],
    deps = [
        ":target_machine_options",
        "//xla:xla_proto_cc",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "constant_allocation",
    srcs = ["constant_allocation.cc"],
    hdrs = ["constant_allocation.h"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_address",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
