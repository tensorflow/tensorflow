load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        "//tensorflow/tsl/lib/io/snappy:__pkg__",
        "//third_party/py/tensorflow_io:__subpackages__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "block",
    hdrs = [
        "block.h",
        "block_builder.h",
        "format.h",
        "table_builder.h",
    ],
    deps = [
        ":iterator",
        ":table_options",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:block",
    ],
)

cc_library(
    name = "buffered_inputstream",
    hdrs = ["buffered_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/lib/io:buffered_inputstream",
    ],
)

cc_library(
    name = "compression",
    hdrs = ["compression.h"],
    deps = ["//tensorflow/tsl/lib/io:compression"],
)

cc_library(
    name = "inputbuffer",
    hdrs = ["inputbuffer.h"],
    deps = [
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:inputbuffer",
    ],
)

cc_library(
    name = "inputstream_interface",
    hdrs = ["inputstream_interface.h"],
    deps = [
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:inputstream_interface",
    ],
)

cc_library(
    name = "iterator",
    hdrs = ["iterator.h"],
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/lib/io:iterator",
    ],
)

cc_library(
    name = "path",
    hdrs = ["path.h"],
    deps = ["//tensorflow/core/platform:path"],
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/lib/io:proto_encode_helper",
    ],
)

cc_library(
    name = "random_inputstream",
    hdrs = ["random_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:env",
        "//tensorflow/tsl/lib/io:random_inputstream",
    ],
)

cc_library(
    name = "record_reader",
    hdrs = ["record_reader.h"],
    deps = [
        ":inputstream_interface",
        ":zlib_compression_options",
        ":zlib_inputstream",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:record_reader",
    ],
)

cc_library(
    name = "record_writer",
    hdrs = ["record_writer.h"],
    deps = [
        ":compression",
        ":zlib_compression_options",
        ":zlib_outputbuffer",
        "//tensorflow/core/lib/hash:crc32c",
        "//tensorflow/core/platform:coding",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:record_writer",
    ],
)

alias(
    name = "snappy_inputbuffer",
    actual = "//tensorflow/tsl/lib/io/snappy:snappy_inputbuffer",
)

alias(
    name = "snappy_inputstream",
    actual = "//tensorflow/tsl/lib/io/snappy:snappy_inputstream",
)

alias(
    name = "snappy_outputbuffer",
    actual = "//tensorflow/tsl/lib/io/snappy:snappy_outputbuffer",
)

alias(
    name = "snappy_compression_options",
    actual = "//tensorflow/tsl/lib/io/snappy:snappy_compression_options",
)

cc_library(
    name = "cache",
    hdrs = ["cache.h"],
    deps = [
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/lib/io:cache",
    ],
)

cc_library(
    name = "table",
    hdrs = [
        "table.h",
        "two_level_iterator.h",
    ],
    deps = [
        ":iterator",
        "//tensorflow/tsl/lib/io:table",
    ],
)

cc_library(
    name = "table_options",
    hdrs = ["table_options.h"],
    deps = ["//tensorflow/tsl/lib/io:table_options"],
)

cc_library(
    name = "zlib_compression_options",
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:zlib_compression_options",
    ],
)

cc_library(
    name = "zlib_inputstream",
    hdrs = ["zlib_inputstream.h"],
    deps = [
        ":inputstream_interface",
        ":zlib_compression_options",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:zlib_inputstream",
    ],
)

cc_library(
    name = "zlib_outputbuffer",
    hdrs = ["zlib_outputbuffer.h"],
    deps = [
        ":zlib_compression_options",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/io:zlib_outputbuffer",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "block.h",
        "block_builder.h",
        "buffered_inputstream.h",
        "cache.h",
        "compression.h",
        "format.h",
        "inputbuffer.h",
        "inputstream_interface.h",
        "iterator.h",
        "path.h",
        "random_inputstream.h",
        "record_reader.h",
        "table.h",
        "table_builder.h",
        "table_options.h",
        "two_level_iterator.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
    ],
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
    name = "legacy_lib_io_headers",
    srcs = [
        "buffered_inputstream.h",
        "cache.h",
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
