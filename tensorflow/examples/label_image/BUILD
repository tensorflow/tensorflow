# Description:
#   Tensorflow C++ inference example for labeling images.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "command_line_flags",
    srcs = [
        "command_line_flags.cc",
        "command_line_flags.h",
    ],
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "label_image",
    srcs = [
        "main.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":command_line_flags",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "command_line_flags_test",
    srcs = [
        "command_line_flags_test.cc",
    ],
    deps = [
        ":command_line_flags",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform/default/build_config:gtest",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
