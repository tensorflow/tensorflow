load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

# Snappy targets.

load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/core/lib/io:__pkg__",
        "//xla/tsl/lib/io:__pkg__",
    ]),
    licenses = ["notice"],
)

exports_files([
    "snappy_compression_options.h",
    "snappy_inputbuffer.h",
    "snappy_inputstream.h",
    "snappy_outputbuffer.h",
    "snappy_inputstream.cc",
    "snappy_test.cc",
])

cc_library(
    name = "snappy_inputbuffer",
    srcs = ["snappy_inputbuffer.cc"],
    hdrs = ["snappy_inputbuffer.h"],
    deps = [
        "//xla/tsl/lib/io:inputstream_interface",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:tstring",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_outputbuffer",
    srcs = ["snappy_outputbuffer.cc"],
    hdrs = ["snappy_outputbuffer.h"],
    deps = [
        "//xla/tsl/lib/io:zlib_compression_options",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:platform_port",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_inputstream",
    srcs = ["snappy_inputstream.cc"],
    hdrs = ["snappy_inputstream.h"],
    deps = [
        "//xla/tsl/lib/io:inputstream_interface",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:tstring",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_compression_options",
    hdrs = ["snappy_compression_options.h"],
    deps = [
        "//xla/tsl/platform:types",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "snappy_test",
    size = "small",
    srcs = ["snappy_test.cc"],
    deps = [
        ":snappy_inputbuffer",
        ":snappy_inputstream",
        ":snappy_outputbuffer",
        "//xla/tsl/lib/io:inputbuffer",
        "//xla/tsl/lib/io:random_inputstream",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:tstring",
    ],
)
