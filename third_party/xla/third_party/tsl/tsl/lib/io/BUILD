load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl:tsl.bzl", "set_external_visibility")
load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "inputbuffer",
    srcs = ["inputbuffer.cc"],
    hdrs = ["inputbuffer.h"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:status",
        "//tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

alias(
    name = "snappy_inputstream",
    actual = "//tsl/lib/io/snappy:snappy_inputstream",
    visibility = ["//visibility:public"],
)

alias(
    name = "snappy_outputbuffer",
    actual = "//tsl/lib/io/snappy:snappy_outputbuffer",
    visibility = ["//visibility:public"],
)

alias(
    name = "snappy_compression_options",
    actual = "//tsl/lib/io/snappy:snappy_compression_options",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cache",
    srcs = [
        "cache.cc",
    ],
    hdrs = [
        "cache.h",
    ],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "block.h",
        "block_builder.h",
        "format.h",
    ],
    visibility = ["//visibility:public"],
)

tsl_cc_test(
    name = "buffered_inputstream_test",
    size = "small",
    srcs = ["buffered_inputstream_test.cc"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
