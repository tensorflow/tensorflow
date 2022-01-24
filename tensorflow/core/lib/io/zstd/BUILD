# Zstd targets.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)
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
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
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
        "//tensorflow/core/lib/io:inputstream_interface",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
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
        "//tensorflow/core/platform:types",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@zstd",
    ],
)
