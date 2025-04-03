load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow/c/experimental/filesystem:__pkg__",
        "//tensorflow/c/experimental/filesystem/plugins/posix:__pkg__",
        "//xla/tsl/lib/io/snappy:__pkg__",
        "//xla:__subpackages__",
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core/util:__subpackages__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/io:__subpackages__",
        "//xla/tsl/profiler:__subpackages__",
        "@local_tsl//tsl/profiler:__subpackages__",
        "//tensorflow/core/profiler:__subpackages__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//third_party/xprof:__subpackages__",
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
        "//xla/tsl/lib/hash:crc32c",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:raw_coding",
        "@local_tsl//tsl/platform:stringpiece",
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
        "//xla/tsl/platform:env",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:coding",
    ],
    alwayslink = True,
)

cc_library(
    name = "inputstream_interface",
    srcs = ["inputstream_interface.cc"],
    hdrs = ["inputstream_interface.h"],
    deps = [
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:cord",
    ],
    alwayslink = True,
)

cc_library(
    name = "iterator",
    srcs = ["iterator.cc"],
    hdrs = ["iterator.h"],
    deps = [
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

cc_library(
    name = "proto_encode_helper",
    hdrs = ["proto_encode_helper.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:stringpiece",
    ],
)

cc_library(
    name = "random_inputstream",
    srcs = ["random_inputstream.cc"],
    hdrs = ["random_inputstream.h"],
    deps = [
        ":inputstream_interface",
        "//xla/tsl/platform:env",
        "@local_tsl//tsl/platform:cord",
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
        "//xla/tsl/lib/hash:crc32c",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:raw_coding",
        "@local_tsl//tsl/platform:stringpiece",
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
        "//xla/tsl/lib/hash:crc32c",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:cord",
        "@local_tsl//tsl/platform:stringpiece",
    ],
    alwayslink = True,
)

alias(
    name = "snappy_inputbuffer",
    actual = "//xla/tsl/lib/io/snappy:snappy_inputbuffer",
)

alias(
    name = "snappy_inputstream",
    actual = "//xla/tsl/lib/io/snappy:snappy_inputstream",
)

alias(
    name = "snappy_outputbuffer",
    actual = "//xla/tsl/lib/io/snappy:snappy_outputbuffer",
)

alias(
    name = "snappy_compression_options",
    actual = "//xla/tsl/lib/io/snappy:snappy_compression_options",
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
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:raw_coding",
        "@local_tsl//tsl/platform:stringpiece",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:coding",
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
        "//xla/tsl/lib/hash:crc32c",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status",
        "@local_tsl//tsl/platform:cord",
    ],
)

tsl_cc_test(
    name = "buffered_file_test",
    size = "small",
    srcs = ["buffered_file_test.cc"],
    deps = [
        ":buffered_file",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "zlib_compression_options",
    srcs = ["zlib_compression_options.cc"],
    hdrs = ["zlib_compression_options.h"],
    deps = [
        "//xla/tsl/platform:types",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:strcat",
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
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@local_tsl//tsl/platform:stringpiece",
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
        "//xla/tsl/lib/io/snappy:snappy_compression_options.h",
        "//xla/tsl/lib/io/snappy:snappy_inputstream.cc",
        "//xla/tsl/lib/io/snappy:snappy_inputstream.h",
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
        "//xla/tsl/lib/io/snappy:snappy_compression_options.h",
        "//xla/tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//xla/tsl/lib/io/snappy:snappy_inputstream.h",
        "//xla/tsl/lib/io/snappy:snappy_outputbuffer.h",
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
        "//xla/tsl/lib/io/snappy:snappy_compression_options.h",
        "//xla/tsl/lib/io/snappy:snappy_inputbuffer.h",
        "//xla/tsl/lib/io/snappy:snappy_inputstream.h",
        "//xla/tsl/lib/io/snappy:snappy_outputbuffer.h",
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
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
    ],
)

tsl_cc_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.cc"],
    deps = [
        ":cache",
        "//xla/tsl/platform:test",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:raw_coding",
    ],
)

tsl_cc_test(
    name = "inputbuffer_test",
    size = "small",
    srcs = ["inputbuffer_test.cc"],
    deps = [
        ":inputbuffer",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:str_util",
        "@local_tsl//tsl/platform:strcat",
    ],
)

tsl_cc_test(
    name = "inputstream_interface_test",
    size = "small",
    srcs = ["inputstream_interface_test.cc"],
    deps = [
        ":inputstream_interface",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "random_inputstream_test",
    size = "small",
    srcs = ["random_inputstream_test.cc"],
    deps = [
        ":random_inputstream",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "record_reader_writer_test",
    size = "small",
    srcs = ["record_reader_writer_test.cc"],
    deps = [
        ":record_reader",
        ":record_writer",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:strcat",
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
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/lib/hash:crc32c",
        "//xla/tsl/lib/random:philox",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:coding",
        "@local_tsl//tsl/platform:str_util",
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
        "//xla/tsl/lib/random:philox",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:platform_port",
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
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:strcat",
    ],
)
