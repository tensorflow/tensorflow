load("//xla/backends/cpu:package_groups.bzl", "xla_cpu_backend_access")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("//xla/tsl/mkl:graph.bzl", "onednn_graph_cc_library")
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
    srcs = ["alignment.h"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "alignment",
    hdrs = ["alignment.h"],
    deps = ["@eigen_archive//:eigen3"],
)

onednn_graph_cc_library(
    name = "onednn_emitter",
    srcs = ["onednn_emitter.cc"],
    hdrs = ["onednn_emitter.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":onednn_fusion",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime/onednn:onednn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@onednn//:mkl_dnn",
    ],
)

onednn_graph_cc_library(
    name = "onednn_fusion",
    srcs = ["onednn_fusion.cc"],
    hdrs = ["onednn_fusion.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_lib",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@onednn//:mkl_dnn",
    ],
)

cc_library(
    name = "xnn_emitter",
    srcs = ["xnn_emitter.cc"],
    hdrs = ["xnn_emitter.h"],
    deps = [
        ":xnn_fusion",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime/xnnpack:xnn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@XNNPACK",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "xnn_fusion",
    srcs = ["xnn_fusion.cc"],
    hdrs = ["xnn_fusion.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/runtime:dot_lib",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
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
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
