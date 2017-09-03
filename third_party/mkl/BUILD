licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE"])

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
        "@mkl//:libmklml_intel.so",
        "@mkl//:libiomp5.so",
    ]),
    visibility = ["//visibility:public"],
    deps = ["@mkl//:mkl_headers"],
)
