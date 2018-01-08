# Build file for the Hlo parser.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [":friends"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
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

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "hlo_lexer",
    srcs = ["hlo_lexer.cc"],
    hdrs = [
        "hlo_lexer.h",
        "hlo_token.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/core:lib",
        "//tensorflow/core:regexp_internal",
    ],
)

cc_library(
    name = "hlo_parser",
    srcs = ["hlo_parser.cc"],
    hdrs = ["hlo_parser.h"],
    deps = [
        ":hlo_lexer",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "hlo_parser_test",
    size = "small",
    srcs = ["hlo_parser_test.cc"],
    deps = [
        ":hlo_parser",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# -----------------------------------------------------------------------------

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
