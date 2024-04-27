load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "proto_serialization",
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    visibility = internal_visibility([
        "@local_xla//xla/pjrt:__subpackages__",
        "@local_xla//xla/python:__pkg__",
        "@local_xla//xla/service:__pkg__",
        "@local_xla//xla/stream_executor:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
        "//tensorflow/compiler/tf2xla/kernels:__pkg__",
        "//tensorflow/core/util/autotune_maps:__pkg__",
    ]),
    deps = [
        "//tsl/lib/gtl:inlined_vector",
        "//tsl/platform:hash",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:protobuf",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "proto_serialization.cc",
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = internal_visibility(["//tensorflow/core/lib/strings:__pkg__"]),
)
