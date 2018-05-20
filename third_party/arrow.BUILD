# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "arrow",
    srcs = [
        "cpp/src/arrow/status.h",
        "cpp/src/arrow/status.cc",
        "cpp/src/arrow/buffer.h",
        "cpp/src/arrow/buffer.cc",
        "cpp/src/arrow/memory_pool.h",
        "cpp/src/arrow/memory_pool.cc",
        "cpp/src/arrow/io/file.h",
        "cpp/src/arrow/io/file.cc",
        "cpp/src/arrow/io/interfaces.h",
        "cpp/src/arrow/io/interfaces.cc",
        "cpp/src/arrow/util/compression.h",
        "cpp/src/arrow/util/compression.cc",
        "cpp/src/arrow/util/key_value_metadata.h",
        "cpp/src/arrow/util/key_value_metadata.cc",
    ],
    includes = [
        "cpp/src/",
    ],
)
