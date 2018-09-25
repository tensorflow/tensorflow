# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")

flatbuffer_cc_library(
    name = "arrow_format",
    srcs = [
				"format/File.fbs",
				"format/Message.fbs",
				"format/Schema.fbs",
				"format/Tensor.fbs",
        "cpp/src/arrow/ipc/feather.fbs",
    ],
    out_prefix = "cpp/src/arrow/ipc/"
)

cc_library(
    name = "arrow",
    srcs = glob([
        "cpp/src/arrow/*.cc",
        "cpp/src/arrow/*.h",
        "cpp/src/arrow/io/*.cc",
        "cpp/src/arrow/io/*.h",
        "cpp/src/arrow/ipc/*.cc",
        "cpp/src/arrow/ipc/*.h",
        "cpp/src/arrow/util/*.cc",
        "cpp/src/arrow/util/*.h",
    ],
    exclude=[
        "cpp/src/arrow/**/*-test.cc",
        "cpp/src/arrow/**/*benchmark*.cc",
        "cpp/src/arrow/**/*hdfs*",
        "cpp/src/arrow/util/compression_zstd.*",
        "cpp/src/arrow/util/compression_lz4.*",
        "cpp/src/arrow/util/compression_brotli.*",
        "cpp/src/arrow/ipc/json*",
        "cpp/src/arrow/ipc/stream-to-file.cc",
        "cpp/src/arrow/ipc/file-to-stream.cc",
    ]),
    hdrs = [
    ],
    defines = [
        "ARROW_WITH_SNAPPY",
    ],
    includes = [
        "cpp/src",
    ],
    copts = [
    ],
    deps = [
        ":arrow_format",
        "@snappy",
    ],
)

