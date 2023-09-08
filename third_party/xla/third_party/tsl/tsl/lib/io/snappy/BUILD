load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

# Snappy targets.

load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    [
        "snappy_compression_options.h",
        "snappy_inputbuffer.h",
        "snappy_inputstream.h",
        "snappy_outputbuffer.h",
        "snappy_inputstream.cc",
        "snappy_test.cc",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "snappy_inputbuffer",
    srcs = ["snappy_inputbuffer.cc"],
    hdrs = ["snappy_inputbuffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/lib/io:inputstream_interface",
        "//tsl/platform:env",
        "//tsl/platform:macros",
        "//tsl/platform:platform_port",
        "//tsl/platform:status",
        "//tsl/platform:types",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_outputbuffer",
    srcs = ["snappy_outputbuffer.cc"],
    hdrs = ["snappy_outputbuffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform",
        "//tsl/platform:env",
        "//tsl/platform:macros",
        "//tsl/platform:platform_port",
        "//tsl/platform:status",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_inputstream",
    srcs = ["snappy_inputstream.cc"],
    hdrs = ["snappy_inputstream.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/lib/io:inputstream_interface",
        "//tsl/platform:errors",
        "//tsl/platform:platform_port",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_compression_options",
    hdrs = ["snappy_compression_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "snappy_test",
    size = "small",
    srcs = ["snappy_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":snappy_inputbuffer",
        ":snappy_inputstream",
        ":snappy_outputbuffer",
        "//tsl/lib/core:status_test_util",
        "//tsl/lib/io:inputbuffer",
        "//tsl/lib/io:random_inputstream",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)
