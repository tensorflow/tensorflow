# Bazel(http://bazel.io) BUILD file

licenses(["notice"])

exports_files(["LICENSE"])

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

cc_library(
    name = "double-conversion",
    srcs = [
        "double-conversion/bignum.cc",
        "double-conversion/bignum-dtoa.cc",
        "double-conversion/cached-powers.cc",
        "double-conversion/diy-fp.cc",
        "double-conversion/double-conversion.cc",
        "double-conversion/fast-dtoa.cc",
        "double-conversion/fixed-dtoa.cc",
        "double-conversion/strtod.cc",
        "double-conversion/utils.h",
    ],
    hdrs = [
        "double-conversion/bignum.h",
        "double-conversion/bignum-dtoa.h",
        "double-conversion/cached-powers.h",
        "double-conversion/diy-fp.h",
        "double-conversion/double-conversion.h",
        "double-conversion/fast-dtoa.h",
        "double-conversion/fixed-dtoa.h",
        "double-conversion/ieee.h",
        "double-conversion/strtod.h",
    ],
    includes = ["."],
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-lm"],
    }),
    visibility = ["//visibility:public"],
)
