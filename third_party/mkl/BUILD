licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like     TODO

config_setting(
    name = "using_mkl",
    values = {
        "define": "using_mkl=true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "intel_binary_blob",
    srcs = [
        "libiomp5.so",
        "libmklml_intel.so",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
