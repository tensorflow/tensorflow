load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        "//tensorflow/tsl/lib/io/snappy:__pkg__",
        "//tensorflow/compiler/xla:__subpackages__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core/util:__subpackages__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/io:__subpackages__",
        "//tensorflow/tsl/profiler:__subpackages__",
    ]),
    licenses = ["notice"],
)

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
        "//tensorflow/tsl/lib/hash:crc32c",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:raw_coding",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:env",
        "@com_google_absl//absl/status",
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
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "inputstream_interface",
    srcs = ["inputstream_interface.cc"],
    hdrs = ["inputstream_interface.h"],
    deps = [
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "iterator",
    srcs = ["iterator.cc"],
    hdrs = ["iterator.h"],
    deps = [
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "random_inputstream",
    srcs = ["random_inputstream.cc"],
    hdrs = ["random_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:env",
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
        "//tensorflow/tsl/lib/hash:crc32c",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:raw_coding",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/lib/hash:crc32c",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
    ],
    alwayslink = True,
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
    srcs = [
        "cache.cc",
    ],
    hdrs = [
        "cache.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:raw_coding",
        "//tensorflow/tsl/platform:stringpiece",
    ],
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
        ":cache",
        ":iterator",
        ":table_options",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
    ],
    alwayslink = True,
)

cc_library(
    name = "table_options",
    hdrs = ["table_options.h"],
)

cc_library(
    name = "buffered_file",
    hdrs = ["buffered_file.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/lib/hash:crc32c",
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "zlib_compression_options",
    srcs = ["zlib_compression_options.cc"],
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:types",
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
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:types",
        "@zlib",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "block.cc",
        "block.h",
        "block_builder.cc",
        "block_builder.h",
        "buffered_file.h",
        "buffered_inputstream.cc",
        "buffered_inputstream.h",
        "cache.cc",
        "cache.h",
        "compression.cc",
        "compression.h",
        "format.cc",
        "format.h",
        "inputbuffer.cc",
        "inputbuffer.h",
        "inputstream_interface.cc",
        "inputstream_interface.h",
        "iterator.cc",
        "iterator.h",
        "random_inputstream.cc",
        "random_inputstream.h",
        "record_reader.cc",
        "record_reader.h",
        "table.cc",
        "table.h",
        "table_builder.cc",
        "table_builder.h",
        "table_options.h",
        "two_level_iterator.cc",
        "two_level_iterator.h",
        "zlib_compression_options.cc",
        "zlib_compression_options.h",
        "zlib_inputstream.cc",
        "zlib_inputstream.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_compression_options.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputstream.cc",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputstream.h",
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
        "//tensorflow/tsl/lib/io/snappy:snappy_compression_options.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputstream.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_outputbuffer.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_io_headers",
    srcs = [
        "buffered_inputstream.h",
        "cache.h",
        "compression.h",
        "inputstream_interface.h",
        "proto_encode_helper.h",
        "random_inputstream.h",
        "record_reader.h",
        "record_writer.h",
        "table.h",
        "table_builder.h",
        "table_options.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "inputbuffer.h",
        "iterator.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
        "zlib_outputbuffer.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_compression_options.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_inputstream.h",
        "//tensorflow/tsl/lib/io/snappy:snappy_outputbuffer.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "block.h",
        "block_builder.h",
        "format.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__pkg__"]),
)

tsl_cc_test(
    name = "buffered_inputstream_test",
    size = "small",
    srcs = ["buffered_inputstream_test.cc"],
    deps = [
        ":buffered_inputstream",
        ":random_inputstream",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.cc"],
    deps = [
        ":cache",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:raw_coding",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "inputbuffer_test",
    size = "small",
    srcs = ["inputbuffer_test.cc"],
    deps = [
        ":inputbuffer",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "inputstream_interface_test",
    size = "small",
    srcs = ["inputstream_interface_test.cc"],
    deps = [
        ":inputstream_interface",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "random_inputstream_test",
    size = "small",
    srcs = ["random_inputstream_test.cc"],
    deps = [
        ":random_inputstream",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "record_reader_writer_test",
    size = "small",
    srcs = ["record_reader_writer_test.cc"],
    deps = [
        ":record_reader",
        ":record_writer",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@zlib",
    ],
)

tsl_cc_test(
    name = "recordio_test",
    size = "small",
    srcs = ["recordio_test.cc"],
    deps = [
        ":record_reader",
        ":record_writer",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/lib/hash:crc32c",
        "//tensorflow/tsl/lib/random:philox",
        "//tensorflow/tsl/platform:coding",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "table_test",
    size = "small",
    srcs = ["table_test.cc"],
    deps = [
        ":block",
        ":iterator",
        ":table",
        "//tensorflow/tsl/lib/random:philox",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "zlib_buffers_test",
    size = "small",
    srcs = ["zlib_buffers_test.cc"],
    deps = [
        ":random_inputstream",
        ":zlib_compression_options",
        ":zlib_inputstream",
        ":zlib_outputbuffer",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)
