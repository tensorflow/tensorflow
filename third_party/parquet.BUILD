# Description:
#   Apache Parquet C++ library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

cc_library(
    name = "parquet",
    srcs = [
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
        "src/parquet/parquet_version.h",
    ],
    hdrs = [
        "src",
    ],
    copts = [
        "-Iexternal/boost",
        "-Iexternal/arrow/cpp/src",
        "-Iexternal/thrift/lib/cpp/src",
    ],
    includes = [
        "src",
    ],
    deps = [
        "@org_tensorflow//third_party/parquet_types",
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
