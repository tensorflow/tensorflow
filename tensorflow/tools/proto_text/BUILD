# Description:
# This package provides build-time generation of proto3 text format functions
# (ProtoDebugString, ProtoShortDebugString, and ProtoParseFromString) which
# provide equivalent functionality as proto.DebugString, proto.ShortDebugString,
# and TextFormat parsing, but can be used with protos generated with
# LITE_RUNTIME.
#
# Note that proto3 well-known types (e.g. Any) are not handled in a special way
# by the generated code.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "placeholder.txt",
])

load(
    "//tensorflow:tensorflow.bzl",
    "if_ios",
    "tf_generate_proto_text_sources",
    "tf_cc_test",
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

cc_binary(
    name = "gen_proto_text_functions",
    srcs = ["gen_proto_text_functions.cc"],
    copts = if_ios(["-DGOOGLE_LOGGING"]),
    visibility = ["//tensorflow:internal"],
    deps = [
        ":gen_proto_text_functions_lib",
        "@protobuf_archive//:protobuf",
        "//tensorflow/core:lib_proto_parsing",
    ] + if_ios(["//tensorflow/core/platform/default/build_config:logging"]),
)

cc_library(
    name = "gen_proto_text_functions_lib",
    srcs = ["gen_proto_text_functions_lib.cc"],
    hdrs = ["gen_proto_text_functions_lib.h"],
    copts = if_ios(["-DGOOGLE_LOGGING"]),
    linkopts = select({
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//tensorflow:darwin": [
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
    ] + if_ios(["//tensorflow/core/platform/default/build_config:logging"]),
)

tf_proto_library_cc(
    name = "test_proto",
    srcs = ["test.proto"],
)

tf_generate_proto_text_sources(
    name = "test_proto_text_srcs",
    srcs = ["test.proto"],
    srcs_relative_dir = "tensorflow/tools/proto_text/",
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

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
