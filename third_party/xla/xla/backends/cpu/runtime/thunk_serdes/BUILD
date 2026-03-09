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
        ":convolution_thunk_serdes",
        ":dot_thunk_serdes",
    ],
    alwayslink = 1,
)
