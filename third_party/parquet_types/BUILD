# Description:
#   Apache Parquet C++ library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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
    ],
    hdrs = [
        "src/parquet/parquet_types.h",
        "src/parquet/util/windows_compatibility.h",
    ],
    includes = [
        "src",
    ],
    copts = [
        "-Iexternal/boost",
        "-Iexternal/arrow/cpp/src",
        "-Iexternal/thrift/lib/cpp/src",
    ],
    deps = [
        "@arrow",
        "@boost",
        "@thrift",
    ],
)
