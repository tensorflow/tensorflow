# Snappy targets.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/core/lib/io:__pkg__",
    ],
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
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/io:inputstream_interface",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_outputbuffer",
    srcs = ["snappy_outputbuffer.cc"],
    hdrs = ["snappy_outputbuffer.h"],
    deps = [
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_inputstream",
    srcs = ["snappy_inputstream.cc"],
    hdrs = ["snappy_inputstream.h"],
    deps = [
        "//tensorflow/core/lib/io:inputstream_interface",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_compression_options",
    hdrs = ["snappy_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)
