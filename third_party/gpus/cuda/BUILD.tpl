licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

load("@local_config_cuda//cuda:platform.bzl", "cuda_library_path")
load("@local_config_cuda//cuda:platform.bzl", "cuda_static_library_path")
load("@local_config_cuda//cuda:platform.bzl", "cudnn_library_path")
load("@local_config_cuda//cuda:platform.bzl", "cupti_library_path")
load("@local_config_cuda//cuda:platform.bzl", "readlink_command")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_gcudacc",
    values = {
        "define": "using_cuda_gcudacc=true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_nvcc",
    values = {
        "define": "using_cuda_nvcc=true",
    },
)

config_setting(
    name = "using_clang",
    values = {
        "define": "using_cuda_clang=true",
    },
)

# Equivalent to using_clang && -c opt.
config_setting(
    name = "using_clang_opt",
    values = {
        "define": "using_cuda_clang=true",
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [
        ".",
        "include",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart_static",
    srcs = [
        cuda_static_library_path("cudart"),
    ],
    includes = ["include/"],
    linkopts = [
        "-ldl",
        "-lpthread",
    ] + select({
        "@//tensorflow:darwin": [],
        "//conditions:default": ["-lrt"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudart",
    srcs = [
        cuda_library_path("cudart"),
    ],
    data = [
        cuda_library_path("cudart"),
    ],
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cublas",
    srcs = [
        cuda_library_path("cublas"),
    ],
    data = [
        cuda_library_path("cublas"),
    ],
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn",
    srcs = [
        cudnn_library_path(),
    ],
    data = [
        cudnn_library_path(),
    ],
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cufft",
    srcs = [
        cuda_library_path("cufft"),
    ],
    data = [
        cuda_library_path("cufft"),
    ],
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "curand",
    srcs = [
        cuda_library_path("curand"),
    ],
    data = [
        cuda_library_path("curand"),
    ],
    includes = ["include/"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cuda",
    deps = [
        ":cuda_headers",
        ":cudart",
        ":cublas",
        ":cudnn",
        ":cufft",
        ":curand",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cupti_headers",
    hdrs = glob([
        "**/*.h",
    ]),
    includes = [
        ".",
        "extras/CUPTI/include/",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cupti_dsos",
    data = [
        cupti_library_path(),
    ],
    visibility = ["//visibility:public"],
)
