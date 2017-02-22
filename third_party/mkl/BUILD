licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like     TODO

config_setting(
    name = "using_mkl",
    values = {
        "define": "using_mkl=true",
    },
    visibility = ["//visibility:public"],
)

load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

cc_library(
    name = "intel_binary_blob",
    srcs = if_mkl([
        "libmklml_intel.so",
        "libiomp5.so",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
