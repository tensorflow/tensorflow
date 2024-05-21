load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        "//tsl/lib/io/snappy:__pkg__",
        "@local_xla//xla:__subpackages__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core/util:__subpackages__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/io:__subpackages__",
        "//tsl/profiler:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
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
        "//tsl/lib/hash:crc32c",
        "//tsl/platform:coding",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:platform_port",
        "//tsl/platform:raw_coding",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
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
        "//tsl/platform:env",
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
        "//tsl/platform:coding",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "inputstream_interface",
    srcs = ["inputstream_interface.cc"],
    hdrs = ["inputstream_interface.h"],
    deps = [
        "//tsl/platform:cord",
        "//tsl/platform:errors",
        "//tsl/platform:status",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

cc_library(
    name = "iterator",
    srcs = ["iterator.cc"],
    hdrs = ["iterator.h"],
    deps = [
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//tsl/platform:coding",
        "//tsl/platform:logging",
        "//tsl/platform:protobuf",
        "//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "random_inputstream",
    srcs = ["random_inputstream.cc"],
    hdrs = ["random_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//tsl/platform:cord",
        "//tsl/platform:env",
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
        "//tsl/lib/hash:crc32c",
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:raw_coding",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
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
        "//tsl/lib/hash:crc32c",
        "//tsl/platform:coding",
        "//tsl/platform:cord",
        "//tsl/platform:env",
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
    ],
    alwayslink = True,
)

alias(
    name = "snappy_inputbuffer",
    actual = "//tsl/lib/io/snappy:snappy_inputbuffer",
)

alias(
    name = "snappy_inputstream",
    actual = "//tsl/lib/io/snappy:snappy_inputstream",
)

alias(
    name = "snappy_outputbuffer",
    actual = "//tsl/lib/io/snappy:snappy_outputbuffer",
)

alias(
    name = "snappy_compression_options",
    actual = "//tsl/lib/io/snappy:snappy_compression_options",
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
        "//tsl/platform:mutex",
        "//tsl/platform:raw_coding",
        "//tsl/platform:stringpiece",
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
        "//tsl/platform:coding",
        "//tsl/platform:env",
        "//tsl/platform:errors",
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
        "//tsl/lib/hash:crc32c",
        "//tsl/platform:cord",
        "//tsl/platform:env",
        "//tsl/platform:status",
    ],
)

tsl_cc_test(
    name = "buffered_file_test",
    size = "small",
    srcs = ["buffered_file_test.cc"],
    deps = [
        ":buffered_file",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "zlib_compression_options",
    srcs = ["zlib_compression_options.cc"],
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//tsl/platform:types",
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
        "//tsl/platform:env",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "//tsl/platform:strcat",
        "//tsl/platform:types",
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
        "//tsl/platform:env",
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
        "//tsl/platform:types",
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
        "//tsl/lib/io/snappy:snappy_compression_options.h",
        "//tsl/lib/io/snappy:snappy_inputstream.cc",
        "//tsl/lib/io/snappy:snappy_inputstream.h",
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
        "//tsl/lib/io/snappy:snappy_compression_options.h",
        "//tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//tsl/lib/io/snappy:snappy_inputstream.h",
        "//tsl/lib/io/snappy:snappy_outputbuffer.h",
    ],
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
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
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "inputbuffer.h",
        "iterator.h",
        "zlib_compression_options.h",
        "zlib_inputstream.h",
        "zlib_outputbuffer.h",
        "//tsl/lib/io/snappy:snappy_compression_options.h",
        "//tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//tsl/lib/io/snappy:snappy_inputstream.h",
        "//tsl/lib/io/snappy:snappy_outputbuffer.h",
    ],
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "block.h",
        "block_builder.h",
        "format.h",
    ],
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

tsl_cc_test(
    name = "buffered_inputstream_test",
    size = "small",
    srcs = ["buffered_inputstream_test.cc"],
    deps = [
        ":buffered_inputstream",
        ":random_inputstream",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.cc"],
    deps = [
        ":cache",
        "//tsl/platform:coding",
        "//tsl/platform:raw_coding",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "inputbuffer_test",
    size = "small",
    srcs = ["inputbuffer_test.cc"],
    deps = [
        ":inputbuffer",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:coding",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:status",
        "//tsl/platform:str_util",
        "//tsl/platform:strcat",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "inputstream_interface_test",
    size = "small",
    srcs = ["inputstream_interface_test.cc"],
    deps = [
        ":inputstream_interface",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:errors",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "random_inputstream_test",
    size = "small",
    srcs = ["random_inputstream_test.cc"],
    deps = [
        ":random_inputstream",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "record_reader_writer_test",
    size = "small",
    srcs = ["record_reader_writer_test.cc"],
    deps = [
        ":record_reader",
        ":record_writer",
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:status",
        "//tsl/platform:strcat",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
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
        "//tsl/lib/core:status_test_util",
        "//tsl/lib/hash:crc32c",
        "//tsl/lib/random:philox",
        "//tsl/platform:coding",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:str_util",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
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
        "//tsl/lib/random:philox",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:platform_port",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
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
        "//tsl/lib/core:status_test_util",
        "//tsl/platform:env",
        "//tsl/platform:env_impl",
        "//tsl/platform:errors",
        "//tsl/platform:strcat",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)
