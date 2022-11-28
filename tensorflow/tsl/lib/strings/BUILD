load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "proto_serialization",
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    visibility = [
        "//tensorflow/compiler/xla/python:__pkg__",
        "//tensorflow/compiler/xla/service:__pkg__",
        "//tensorflow/compiler/xla/stream_executor:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
    ],
    deps = [
        "//tensorflow/tsl/lib/gtl:inlined_vector",
        "//tensorflow/tsl/platform:hash",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
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
    visibility = ["//tensorflow/core/lib/strings:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//tensorflow/core/lib/strings:__pkg__"],
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//tensorflow/core/lib/strings:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//tensorflow/core/lib/strings:__pkg__"],
)
