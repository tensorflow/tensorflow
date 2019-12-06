package(
    default_visibility = [
        "//tensorflow/core/lib/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
        "//tensorflow/core/lib/hash:__pkg__",
        "//tensorflow/core/lib/io:__pkg__",
        "//tensorflow/core/lib/random:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are:
# block, block_builder, buffered_inputstream, format, inputbuffer,
# random_inputstream, record_reader, record_writer, snappy/snappy_inputbuffer
# snappy/snappy_outputbuffer, table, table_builder, two_level_iterator,
# zlib_inputstream, zlib_outputbuffer, zlib_compression_options, and all tests.

cc_library(
    name = "compression",
    srcs = ["compression.cc"],
    hdrs = ["compression.h"],
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
)

cc_library(
    name = "iterator",
    srcs = ["iterator.cc"],
    hdrs = ["iterator.h"],
    deps = [
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/core:stringpiece",
    ],
)

cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/lib/strings:string_utils",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/platform:protobuf",
    ],
)

cc_library(
    name = "table_options",
    hdrs = ["table_options.h"],
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
        "path.cc",
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
