# Zstd targets.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/tsl/lib/io:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files([
    "zstd_compression_options.h",
    "zstd_inputstream.h",
    "zstd_outputbuffer.h",
    "zstd_inputstream.cc",
    "zstd_test.cc",
])

cc_library(
    name = "zstd_outputbuffer",
    srcs = ["zstd_outputbuffer.cc"],
    hdrs = ["zstd_outputbuffer.h"],
    deps = [
        ":zstd_compression_options",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
        "@zstd",
    ],
    alwayslink = True,
)

cc_library(
    name = "zstd_inputstream",
    srcs = ["zstd_inputstream.cc"],
    hdrs = ["zstd_inputstream.h"],
    deps = [
        ":zstd_compression_options",
        "//tensorflow/tsl/lib/io:inputstream_interface",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/memory",
        "@zstd",
    ],
    alwayslink = True,
)

cc_library(
    name = "zstd_compression_options",
    srcs = ["zstd_compression_options.cc"],
    hdrs = ["zstd_compression_options.h"],
    deps = [
        "//tensorflow/tsl/platform:types",
        "@zstd",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "zstd_test",
    srcs = ["zstd_test.cc"],
    deps = [
        ":zstd_compression_options",
        ":zstd_inputstream",
        ":zstd_outputbuffer",
        "//tensorflow/tsl:lib",
        "//tensorflow/tsl:lib_internal",
        "//tensorflow/tsl:test",
        "//tensorflow/tsl:test_main",
        "//tensorflow/tsl:testlib",
        "@zstd",
    ],
)
