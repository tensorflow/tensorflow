load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "leveldb",
    srcs = [],
    hdrs = ["leveldb.h"],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/util/tensor_bundle:__pkg__",
    ],
    deps = [
        "//tensorflow/core/platform:env",
        "@leveldb",
    ],
    alwayslink = True,
)

cc_library(
    name = "buffered_inputstream",
    srcs = ["buffered_inputstream.cc"],
    hdrs = ["buffered_inputstream.h"],
    deps = [
        ":inputstream_interface",
        ":random_inputstream",
        "//tensorflow/core/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "compression",
    srcs = ["compression.cc"],
    hdrs = ["compression.h"],
    alwayslink = True,
)

cc_library(
    name = "inputbuffer",
    srcs = ["inputbuffer.cc"],
    hdrs = ["inputbuffer.h"],
    deps = [
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "inputstream_interface",
    srcs = ["inputstream_interface.cc"],
    hdrs = ["inputstream_interface.h"],
    deps = [
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    deps = [
        "//tensorflow/core/platform:path",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:stringpiece",
    ],
)

cc_library(
    name = "random_inputstream",
    srcs = ["random_inputstream.cc"],
    hdrs = ["random_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "record_reader",
    srcs = ["record_reader.cc"],
    hdrs = ["record_reader.h"],
    deps = [
        ":buffered_inputstream",
        ":compression",
        ":inputstream_interface",
        ":random_inputstream",
        ":snappy_compression_options",
        ":snappy_inputstream",
        ":zlib_compression_options",
        ":zlib_inputstream",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/hash:crc32c",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "record_writer",
    srcs = ["record_writer.cc"],
    hdrs = ["record_writer.h"],
    deps = [
        ":compression",
        ":snappy_compression_options",
        ":snappy_outputbuffer",
        ":zlib_compression_options",
        ":zlib_outputbuffer",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/hash:crc32c",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_inputbuffer",
    srcs = ["snappy/snappy_inputbuffer.cc"],
    hdrs = ["snappy/snappy_inputbuffer.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_outputbuffer",
    srcs = ["snappy/snappy_outputbuffer.cc"],
    hdrs = ["snappy/snappy_outputbuffer.h"],
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
    srcs = ["snappy/snappy_inputstream.cc"],
    hdrs = ["snappy/snappy_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = True,
)

cc_library(
    name = "snappy_compression_options",
    hdrs = ["snappy/snappy_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "zlib_compression_options",
    srcs = ["zlib_compression_options.cc"],
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
        "@zlib",
    ],
    alwayslink = True,
)

cc_library(
    name = "zlib_inputstream",
    srcs = ["zlib_inputstream.cc"],
    hdrs = ["zlib_inputstream.h"],
    deps = [
        ":inputstream_interface",
        ":zlib_compression_options",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:strcat",
        "//tensorflow/core/platform:types",
        "@zlib",
    ],
    alwayslink = True,
)

cc_library(
    name = "zlib_outputbuffer",
    srcs = ["zlib_outputbuffer.cc"],
    hdrs = ["zlib_outputbuffer.h"],
    deps = [
        ":zlib_compression_options",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@zlib",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "buffered_inputstream.cc",
        "buffered_inputstream.h",
        "compression.cc",
        "compression.h",
        "inputbuffer.cc",
        "inputbuffer.h",
        "inputstream_interface.cc",
        "inputstream_interface.h",
        "leveldb.h",
        "path.h",
        "random_inputstream.cc",
        "random_inputstream.h",
        "record_reader.cc",
        "record_reader.h",
        "snappy/snappy_compression_options.h",
        "snappy/snappy_inputstream.cc",
        "snappy/snappy_inputstream.h",
        "zlib_compression_options.cc",
        "zlib_compression_options.h",
        "zlib_inputstream.cc",
        "zlib_inputstream.h",
    ],
)

filegroup(
    name = "legacy_lib_io_all_headers",
    srcs = [
        "buffered_inputstream.h",
        "compression.h",
        "inputbuffer.h",
        "inputstream_interface.h",
        "path.h",
        "proto_encode_helper.h",
        "random_inputstream.h",
        "record_reader.h",
        "record_writer.h",
        "snappy/snappy_compression_options.h",
        "snappy/snappy_inputbuffer.h",
        "snappy/snappy_inputstream.h",
        "snappy/snappy_outputbuffer.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
        "zlib_outputbuffer.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_io_all_tests",
    srcs = [
        "buffered_inputstream_test.cc",
        "inputbuffer_test.cc",
        "inputstream_interface_test.cc",
        "path_test.cc",
        "random_inputstream_test.cc",
        "record_reader_writer_test.cc",
        "recordio_test.cc",
        "snappy/snappy_test.cc",
        "zlib_buffers_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_io_headers",
    srcs = [
        "buffered_inputstream.h",
        "compression.h",
        "inputstream_interface.h",
        "path.h",
        "proto_encode_helper.h",
        "random_inputstream.h",
        "record_reader.h",
        "record_writer.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "inputbuffer.h",
        "snappy/snappy_compression_options.h",
        "snappy/snappy_inputbuffer.h",
        "snappy/snappy_inputstream.h",
        "snappy/snappy_outputbuffer.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
        "zlib_outputbuffer.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
