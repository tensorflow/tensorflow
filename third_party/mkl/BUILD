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

    srcs = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "@mkl_linux//:libmklml_intel.so",
            "@mkl_linux//:libiomp5.so",
        ],
        "@org_tensorflow//tensorflow:darwin": [
            "@mkl_darwin//:libmklml_intel.so",
            "@mkl_darwin//:libiomp5.so",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "@mkl_windows//:libmklml_intel.so",
            "@mkl_windows//:libiomp5.so",
        ]
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@org_tensorflow//tensorflow:linux_x86_64": [
            "@mkl_linux//:mkl_headers",
        ],
        "@org_tensorflow//tensorflow:darwin": [
            "@mkl_darwin//:mkl_headers",
        ],
        "@org_tensorflow//tensorflow:windows": [
            "@mkl_windows//:mkl_headers",
        ]
    })
    # ["@mkl//:mkl_headers"],
)
