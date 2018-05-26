# Description:
#   Apache Parquet C++ library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

cc_library(
    name = "parquet",
    srcs = [
        "src/parquet/api/io.h",
        "src/parquet/api/reader.h",
        "src/parquet/api/schema.h",
        "src/parquet/encoding.h",
        "src/parquet/encoding-internal.h",
        "src/parquet/schema-internal.h",
        "src/parquet/thrift.h",
        "src/parquet/properties.h",
        "src/parquet/column_page.h",
        "src/parquet/column_reader.cc",
        "src/parquet/column_reader.h",
        "src/parquet/column_scanner.cc",
        "src/parquet/column_scanner.h",
        "src/parquet/exception.cc",
        "src/parquet/exception.h",
        "src/parquet/file_reader.cc",
        "src/parquet/file_reader.h",
        "src/parquet/metadata.cc",
        "src/parquet/metadata.h",
        "src/parquet/printer.cc",
        "src/parquet/printer.h",
        "src/parquet/schema.cc",
        "src/parquet/schema.h",
        "src/parquet/statistics.cc",
        "src/parquet/statistics.h",
        "src/parquet/types.cc",
        "src/parquet/types.h",
        "src/parquet/util/comparison.cc",
        "src/parquet/util/comparison.h",
        "src/parquet/util/memory.cc",
        "src/parquet/util/memory.h",
        "src/parquet/util/visibility.h",
        "src/parquet/util/macros.h",
        "src/parquet/util/logging.h",
        "src/parquet/util/windows_compatibility.h",
        "src/parquet/parquet_version.h",
        "src/parquet/parquet_types.cpp",
        "src/parquet/parquet_types.h",
    ],
    hdrs = [
    ],
    copts = [
    ],
    includes = [
        "src",
    ],
    deps = [
        "@arrow",
        "@boost",
        "@thrift",
    ],
)

template_rule(
    name = "parquet_version_h",
    src = "src/parquet/parquet_version.h.in",
    out = "src/parquet/parquet_version.h",
    substitutions = {
        "@PARQUET_VERSION@": "1.4.0",
    },
)
