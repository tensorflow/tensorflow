licenses(["notice"])  # 3-Clause BSD

config_setting(
    name = "using_mkl",
    values = {
        "define": "using_mkl=true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_mkl_lnx_x64",
    values = {
        "cpu": "k8",
        "define": "using_mkl=true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_mkl_darwin",
    values = {
        "cpu": "darwin",
        "define": "using_mkl=true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSE",
    srcs = [
        "MKL_LICENSE",
        "@mkl//:LICENSE"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "intel_binary_blob",
    srcs = [
        "@mkl//:libmklml",
        "@mkl//:libiomp5",
    ],
    hdrs = ["@mkl//:mkl_headers"],
    strip_include_prefix = "/external/mkl/include",
    visibility = ["//visibility:public"],
)
