# Description:
#   Apache Arrow library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

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
        "cpp/src/arrow/util/compression_snappy.h",
        "cpp/src/arrow/util/compression_snappy.cc",
        "cpp/src/arrow/util/key_value_metadata.h",
        "cpp/src/arrow/util/key_value_metadata.cc",
        "cpp/src/arrow/util/visibility.h",
        "cpp/src/arrow/util/macros.h",
        "cpp/src/arrow/util/bit-util.h",
        "cpp/src/arrow/util/type_traits.h",
        "cpp/src/arrow/io/memory.h",
        "cpp/src/arrow/util/logging.h",
        "cpp/src/arrow/builder.h",
        "cpp/src/arrow/type.h",
        "cpp/src/arrow/type_fwd.h",
        "cpp/src/arrow/visitor.h",
        "cpp/src/arrow/type_traits.h",
        "cpp/src/arrow/util/hash.h",
        "cpp/src/arrow/util/bit-stream-utils.h",
        "cpp/src/arrow/util/bpacking.h",
        "cpp/src/arrow/util/cpu-info.h",
        "cpp/src/arrow/util/hash-util.h",
        "cpp/src/arrow/util/sse-util.h",
        "cpp/src/arrow/util/rle-encoding.h",
    ],
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
        "@snappy",
    ],
)
