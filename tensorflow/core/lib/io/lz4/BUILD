# lz4 targets.

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
    "lz4_compression_options.h",
    "lz4_inputstream.h",
    "lz4_outputbuffer.h",
    "lz4_inputstream.cc",
    "lz4_test.cc",
])

cc_library(
    name = "lz4_outputbuffer",
    srcs = ["lz4_outputbuffer.cc"],
    hdrs = ["lz4_outputbuffer.h"],
    deps = [
        ":lz4_compression_options",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@lz4",
    ],
    alwayslink = True,
)

cc_library(
    name = "lz4_inputstream",
    srcs = ["lz4_inputstream.cc"],
    hdrs = ["lz4_inputstream.h"],
    deps = [
        ":lz4_compression_options",
        "//tensorflow/core/lib/io:inputstream_interface",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/memory",
        "@lz4",
    ],
    alwayslink = True,
)

cc_library(
    name = "lz4_compression_options",
    srcs = ["lz4_compression_options.cc"],
    hdrs = ["lz4_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
        "@lz4",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "lz4_test",
    srcs = ["lz4_test.cc"],
    deps = [
        ":lz4_compression_options",
        ":lz4_inputstream",
        ":lz4_outputbuffer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@lz4",
    ],
)
