# Description:
#   Apache Parquet C++ library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@org_tensorflow//third_party:common.bzl", "template_rule")

# Note: parquet_types.cpp/parquet_types.h
# are pre-generated statically. The reason
# to not generate during the tf build is
# that they depend on bison/flex, and so many
# other tools. It is better to generate them
# copy here.
# Also windows_compatibility.h is needed by
# parquet_types.cpp.
cc_library(
    name = "parquet_types",
    srcs = [
        "src/parquet/parquet_types.cpp",
        "src/parquet/parquet_types.h",
        "src/parquet/util/windows_compatibility.h",
    ],
    hdrs = [
        "src/parquet/parquet_types.h",
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
    ],
    hdrs = [
        "src/parquet/parquet_version.h",
    ],
    copts = [
    ],
    includes = [
        "src/",
    ],
    deps = [
        "@org_tensorflow//third_party/parquet:parquet_types",
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
