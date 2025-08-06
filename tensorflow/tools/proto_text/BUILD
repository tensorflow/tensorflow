# Description:
# This package provides build-time generation of proto3 text format functions
# (ProtoDebugString, ProtoShortDebugString, and ProtoParseFromString) which
# provide equivalent functionality as proto.DebugString, proto.ShortDebugString,
# and TextFormat parsing, but can be used with protos generated with
# LITE_RUNTIME.
#
# Note that proto3 well-known types (e.g. Any) are not handled in a special way
# by the generated code.

load(
    "//tensorflow:tensorflow.bzl",
    "if_ios",
    "tf_cc_test",
    "tf_generate_proto_text_sources",
)

# For platform specific build config
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

exports_files(["placeholder.txt"])

cc_binary(
    name = "gen_proto_text_functions",
    srcs = ["gen_proto_text_functions.cc"],
    copts = if_ios(["-DGOOGLE_LOGGING"]),
    visibility = ["//tensorflow:internal"],
    deps = [
        ":gen_proto_text_functions_lib",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@local_xla//xla/tsl/platform:protobuf_compiler",
    ] + if_ios(["//tensorflow/core/platform:logging"]),
)

cc_library(
    name = "gen_proto_text_functions_lib",
    srcs = ["gen_proto_text_functions_lib.cc"],
    hdrs = ["gen_proto_text_functions_lib.h"],
    copts = if_ios(["-DGOOGLE_LOGGING"]),
    linkopts = select({
        "//tensorflow:windows": [],
        "//tensorflow:macos": [
            "-lm",
            "-lpthread",
        ],
        "//tensorflow:ios": [
            "-lm",
            "-lpthread",
        ],
        "//conditions:default": [
            "-lm",
            "-lpthread",
            "-lrt",
        ],
    }),
    deps = [
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/strings:string_view",
    ] + if_ios(["//tensorflow/core/platform:logging"]),
)

tf_proto_library(
    name = "test_proto",
    srcs = ["test.proto"],
)

tf_generate_proto_text_sources(
    name = "test_proto_text",
    srcs = ["test.proto"],
    srcs_relative_dir = "tensorflow/tools/proto_text/",
    deps = [
        ":test_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "gen_proto_text_functions_lib_test",
    size = "small",
    srcs = [
        "gen_proto_text_functions_lib_test.cc",
        ":test_proto_text_srcs",
    ],
    deps = [
        ":gen_proto_text_functions_lib",
        ":test_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
