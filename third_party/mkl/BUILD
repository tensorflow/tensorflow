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
    includes = ["."],
    srcs = ["libmklml_intel.so", "libiomp5.so"],
    visibility = ["//visibility:public"],
)
