load(":build_defs.bzl", "cuda_header_library")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

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
)

config_setting(
    name = "freebsd",
    values = {"cpu": "freebsd"},
)

cuda_header_library(
    name = "cuda_headers",
    hdrs = [
        "cuda/cuda_config.h",
        ":cuda-include"
    ],
    include_prefix = "third_party/gpus",
    includes = [
        ".",  # required to include cuda/cuda/cuda_config.h as cuda/config.h
        "cuda/include",
    ],
)

cc_library(
    name = "cudart_static",
    srcs = ["cuda/lib/%{cudart_static_lib}"],
    linkopts = select({
        ":freebsd": [],
        "//conditions:default": ["-ldl"],
    }) + [
        "-lpthread",
        %{cudart_static_linkopt}
    ],
)

cc_library(
    name = "cuda_driver",
    srcs = ["cuda/lib/%{cuda_driver_lib}"],
)

cc_library(
    name = "cudart",
    srcs = ["cuda/lib/%{cudart_lib}"],
    data = ["cuda/lib/%{cudart_lib}"],
    linkstatic = 1,
)

cuda_header_library(
    name = "cublas_headers",
    hdrs = [":cublas-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cublas/include",
    deps = [":cuda_headers"],
    includes = ["cublas/include"],
)

cuda_header_library(
    name = "cusolver_headers",
    hdrs = [":cusolver-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cusolver/include",
    deps = [":cuda_headers"],
    includes = ["cusolver/include"],
)

cuda_header_library(
    name = "cufft_headers",
    hdrs = [":cufft-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cufft/include",
    deps = [":cuda_headers"],
    includes = ["cufft/include"],
)

cuda_header_library(
    name = "cusparse_headers",
    hdrs = [":cusparse-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "cusparse/include",
    deps = [":cuda_headers"],
    includes = ["cusparse/include"],
)

cuda_header_library(
    name = "curand_headers",
    hdrs = [":curand-include"],
    include_prefix = "third_party/gpus/cuda/include",
    strip_include_prefix = "curand/include",
    deps = [":cuda_headers"],
    includes = ["curand/include"],
)

cc_library(
    name = "cublas",
    srcs = ["cuda/lib/%{cublas_lib}"],
    data = ["cuda/lib/%{cublas_lib}"],
    linkstatic = 1,
)

cc_library(
    name = "cusolver",
    srcs = ["cuda/lib/%{cusolver_lib}"],
    data = ["cuda/lib/%{cusolver_lib}"],
    linkopts = ["-lgomp"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn",
    srcs = ["cuda/lib/%{cudnn_lib}"],
    data = ["cuda/lib/%{cudnn_lib}"],
    linkstatic = 1,
)

cc_library(
    name = "cudnn_header",
    hdrs = [":cudnn-include"],
    include_prefix = "third_party/gpus/cudnn",
    strip_include_prefix = "cudnn/include",
    deps = [":cuda_headers"],
)

cc_library(
    name = "cufft",
    srcs = ["cuda/lib/%{cufft_lib}"],
    data = ["cuda/lib/%{cufft_lib}"],
    linkstatic = 1,
)

cc_library(
    name = "curand",
    srcs = ["cuda/lib/%{curand_lib}"],
    data = ["cuda/lib/%{curand_lib}"],
    linkstatic = 1,
)

cc_library(
    name = "cuda",
    deps = [
        ":cublas",
        ":cuda_headers",
        ":cudart",
        ":cudnn",
        ":cufft",
        ":curand",
    ],
)

cuda_header_library(
    name = "cupti_headers",
    hdrs = [":cuda-extras"],
    include_prefix="third_party/gpus",
    includes = ["cuda/extras/CUPTI/include/"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cupti_dsos",
    data = ["cuda/lib/%{cupti_lib}"],
)

cc_library(
    name = "cusparse",
    srcs = ["cuda/lib/%{cusparse_lib}"],
    data = ["cuda/lib/%{cusparse_lib}"],
    linkopts = ["-lgomp"],
    linkstatic = 1,
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
)

filegroup(
    name = "cuda_root",
    srcs = [
        "cuda/bin/fatbinary",
        "cuda/bin/bin2c",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "@bazel_skylib//lib:selects",
    ],
)

py_library(
    name = "cuda_config_py",
    srcs = ["cuda/cuda_config.py"]
)

%{copy_rules}
