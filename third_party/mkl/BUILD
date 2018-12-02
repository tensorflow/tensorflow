licenses(["notice"])  # 3-Clause BSD

config_setting(
    name = "build_with_mkl",
    define_values = {
        "build_with_mkl": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_ml_only",
    define_values = {
        "build_with_mkl": "true",
        "build_with_mkl_ml_only": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "build_with_mkl_lnx_x64",
    define_values = {
        "build_with_mkl": "true",
    },
    values = {
        "cpu": "k8",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "enable_mkl",
    define_values = {
        "enable_mkl": "true",
        "build_with_mkl": "true",
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
        "//conditions:default": [],
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
        "//conditions:default": [],
    }),
)
