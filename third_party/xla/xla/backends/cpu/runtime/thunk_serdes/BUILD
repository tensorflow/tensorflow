load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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

cc_library(
    name = "convolution_thunk_serdes",
    srcs = ["convolution_thunk_serdes.cc"],
    hdrs = ["convolution_thunk_serdes.h"],
    deps = [
        "//xla/backends/cpu/runtime:convolution_dims",
        "//xla/backends/cpu/runtime:convolution_thunk",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_impl",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_utils",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:resource_use",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:casts",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dot_thunk_serdes",
    srcs = ["dot_thunk_serdes.cc"],
    hdrs = ["dot_thunk_serdes.h"],
    deps = [
        "//xla/backends/cpu/runtime:dot_thunk",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_impl",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_utils",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:resource_use",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:casts",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ynn_fusion_thunk_serdes",
    srcs = ["ynn_fusion_thunk_serdes.cc"],
    hdrs = ["ynn_fusion_thunk_serdes.h"],
    deps = [
        "//xla:util",
        "//xla/backends/cpu:ynn_emitter",
        "//xla/backends/cpu:ynn_fusion_options_proto_cc",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_impl",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_utils",
        "//xla/backends/cpu/runtime/ynnpack:ynn_fusion_thunk",
        "//xla/backends/cpu/runtime/ynnpack:ynn_interop",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:resource_use",
        "//xla/service:buffer_assignment",
        "//xla/stream_executor:device_address",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:casts",
    ],
    alwayslink = 1,
)

cc_library(
    name = "collective_thunk_serdes",
    srcs = ["collective_thunk_serdes.cc"],
    hdrs = ["collective_thunk_serdes.h"],
    deps = [
        "//xla:util",
        "//xla/backends/cpu/runtime:all_gather_thunk",
        "//xla/backends/cpu/runtime:all_reduce_thunk",
        "//xla/backends/cpu/runtime:all_to_all_thunk",
        "//xla/backends/cpu/runtime:collective_permute_thunk",
        "//xla/backends/cpu/runtime:collective_thunk",
        "//xla/backends/cpu/runtime:reduce_scatter_thunk",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_impl",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_utils",
        "//xla/core/collectives:reduction_kind",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:resource_use",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:casts",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fft_thunk_serdes",
    srcs = ["fft_thunk_serdes.cc"],
    hdrs = ["fft_thunk_serdes.h"],
    deps = [
        "//xla/backends/cpu/runtime:fft_thunk",
        "//xla/backends/cpu/runtime:thunk",
        "//xla/backends/cpu/runtime:thunk_proto_cc",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_impl",
        "//xla/backends/cpu/runtime:thunk_proto_serdes_utils",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:resource_use",
        "//xla/service:buffer_assignment",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:casts",
    ],
    alwayslink = 1,
)

# Utility target to link all thunk serdes implementations.
cc_library(
    name = "all_thunk_serdes",
    deps = [
        ":collective_thunk_serdes",
        ":convolution_thunk_serdes",
        ":dot_thunk_serdes",
        ":fft_thunk_serdes",
        ":ynn_fusion_thunk_serdes",
    ],
    alwayslink = 1,
)
