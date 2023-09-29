load("//tsl:tsl.bzl", "set_external_visibility")
load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "proto_serialization",
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "proto_serialization.h",
    ],
    visibility = ["//visibility:public"],
)
