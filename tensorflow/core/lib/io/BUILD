package(
    default_visibility = [
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "block",
    srcs = [
        "block.cc",
        "block_builder.cc",
        "format.cc",
        "table_builder.cc",
    ],
    hdrs = [
        "block.h",
        "block_builder.h",
        "format.h",
        "table_builder.h",
    ],
    deps = [
        ":iterator",
        ":table_options",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/hash:crc32c",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/platform:types",
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
    name = "iterator",
    srcs = ["iterator.cc"],
    hdrs = ["iterator.h"],
    deps = [
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:stringpiece",
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
    name = "table",
    srcs = [
        "table.cc",
        "two_level_iterator.cc",
    ],
    hdrs = [
        "table.h",
        "two_level_iterator.h",
    ],
    deps = [
        ":block",
        ":iterator",
        ":table_options",
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/platform:env",
    ],
    alwayslink = True,
)

cc_library(
    name = "table_options",
    hdrs = ["table_options.h"],
)

cc_library(
    name = "zlib_compression_options",
    srcs = ["zlib_compression_options.cc"],
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
        "@zlib_archive//:zlib",
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
        "@zlib_archive//:zlib",
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
        "@zlib_archive//:zlib",
    ],
    alwayslink = True,
)

filegroup(
    name = "legacy_lib_io_all_headers",
    srcs = [
        "block.h",
        "block_builder.h",
        "buffered_inputstream.h",
        "compression.h",
        "format.h",
        "inputbuffer.h",
        "inputstream_interface.h",
        "iterator.h",
        "path.h",
        "proto_encode_helper.h",
        "random_inputstream.h",
        "record_reader.h",
        "record_writer.h",
        "snappy/snappy_inputbuffer.h",
        "snappy/snappy_outputbuffer.h",
        "table.h",
        "table_builder.h",
        "table_options.h",
        "two_level_iterator.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
        "zlib_outputbuffer.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_io_all_srcs",
    srcs = [
        "block.cc",
        "block_builder.cc",
        "buffered_inputstream.cc",
        "compression.cc",
        "format.cc",
        "inputbuffer.cc",
        "inputstream_interface.cc",
        "iterator.cc",
        "random_inputstream.cc",
        "record_reader.cc",
        "record_writer.cc",
        "snappy/snappy_inputbuffer.cc",
        "snappy/snappy_outputbuffer.cc",
        "table.cc",
        "table_builder.cc",
        "two_level_iterator.cc",
        "zlib_compression_options.cc",
        "zlib_inputstream.cc",
        "zlib_outputbuffer.cc",
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
        "snappy/snappy_buffers_test.cc",
        "table_test.cc",
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
        "table.h",
        "table_builder.h",
        "table_options.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "inputbuffer.h",
        "iterator.h",
        "snappy/snappy_inputbuffer.h",
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
        "block.h",
        "block_builder.h",
        "format.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
