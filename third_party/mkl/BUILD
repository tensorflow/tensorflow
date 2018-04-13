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

load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)

filegroup(
    name = "LICENSE",
    srcs = ["MKL_LICENSE"] + select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "@mkl_linux//:LICENSE",
        ],
        "@org_tensorflow//tensorflow:darwin": [
            "@mkl_darwin//:LICENSE",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "@mkl_windows//:LICENSE",
        ],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "intel_binary_blob",
    visibility = ["//visibility:public"],
    deps = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "@mkl_linux//:mkl_headers",
            "@mkl_linux//:mkl_libs_linux",
        ],
        "@org_tensorflow//tensorflow:darwin": [
            "@mkl_darwin//:mkl_headers",
            "@mkl_darwin//:mkl_libs_darwin",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "@mkl_windows//:mkl_headers",
            "@mkl_windows//:mkl_libs_windows",
        ],
    }),
)
