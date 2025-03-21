# NB: DEPRECATED! This file is a part of the deprecated `cuda_configure` rule.
# Hermetic CUDA repository rule doesn't support Windows.
# Please use `hermetic/cuda_configure`.

load(":build_defs.bzl", "cuda_header_library")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

# Config setting whether TensorFlow is built with CUDA support using clang.
#
# TODO(b/174244321), DEPRECATED: this target will be removed when all users
# have been converted to :is_cuda_enabled (most) or :is_cuda_compiler_clang.
selects.config_setting_group(
    name = "using_clang",
    match_all = [
        "@local_config_cuda//:is_cuda_enabled",
        "@local_config_cuda//:is_cuda_compiler_clang",
    ],
)

# Config setting whether TensorFlow is built with CUDA support using nvcc.
#
# TODO(b/174244321), DEPRECATED: this target will be removed when all users
# have been converted to :is_cuda_enabled (most) or :is_cuda_compiler_nvcc.
selects.config_setting_group(
    name = "using_nvcc",
    match_all = [
        "@local_config_cuda//:is_cuda_enabled",
        "@local_config_cuda//:is_cuda_compiler_nvcc",
    ],
)

# Equivalent to using_clang && -c opt.
selects.config_setting_group(
    name = "using_clang_opt",
    match_all = [
        ":using_clang",
        ":_opt",
    ],
)

config_setting(
    name = "_opt",
    values = {"compilation_mode": "opt"},
)

# Provides CUDA headers for '#include "third_party/gpus/cuda/include/cuda.h"'
# All clients including TensorFlow should use these directives.
cuda_header_library(
    name = "cuda_headers",
    hdrs = [
        "cuda/cuda_config.h",
        ":cuda-include",
    ],
    include_prefix = "third_party/gpus",
    includes = [
        ".",  # required to include cuda/cuda/cuda_config.h as cuda/config.h
        "cuda/include",
    ],
)

cc_import(
    name = "cudart_static",
    # /WHOLEARCHIVE:cudart_static.lib will cause a
    # "Internal error during CImplib::EmitThunk" error.
    # Treat this library as interface library to avoid being whole archived when
    # linking a DLL that depends on this.
    # TODO(pcloudy): Remove this rule after b/111278841 is resolved.
    interface_library = "cuda/lib/%{cudart_static_lib}",
    system_provided = 1,
)

cc_import(
    name = "cuda_driver",
    interface_library = "cuda/lib/%{cuda_driver_lib}",
    system_provided = 1,
)

cc_import(
    name = "cudart",
    interface_library = "cuda/lib/%{cudart_lib}",
    system_provided = 1,
)

cuda_header_library(
    name = "cublas_headers",
    hdrs = [":cublas-include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["cublas/include"],
    strip_include_prefix = "cublas/include",
    deps = [":cuda_headers"],
)

cuda_header_library(
    name = "cusolver_headers",
    hdrs = [":cusolver-include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["cusolver/include"],
    strip_include_prefix = "cusolver/include",
    deps = [":cuda_headers"],
)

cuda_header_library(
    name = "cufft_headers",
    hdrs = [":cufft-include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["cufft/include"],
    strip_include_prefix = "cufft/include",
    deps = [":cuda_headers"],
)

cuda_header_library(
    name = "cusparse_headers",
    hdrs = [":cusparse-include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["cusparse/include"],
    strip_include_prefix = "cusparse/include",
    deps = [":cuda_headers"],
)

cuda_header_library(
    name = "curand_headers",
    hdrs = [":curand-include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["curand/include"],
    strip_include_prefix = "curand/include",
    deps = [":cuda_headers"],
)

cc_import(
    name = "cublas",
    interface_library = "cuda/lib/%{cublas_lib}",
    system_provided = 1,
)

cc_import(
    name = "cublasLt",
    interface_library = "cuda/lib/%{cublasLt_lib}",
    system_provided = 1,
)

cc_import(
    name = "cusolver",
    interface_library = "cuda/lib/%{cusolver_lib}",
    system_provided = 1,
)

cc_import(
    name = "cudnn",
    interface_library = "cuda/lib/%{cudnn_lib}",
    system_provided = 1,
)

cc_library(
    name = "cudnn_header",
    hdrs = [":cudnn-include"],
    include_prefix = "third_party/gpus/cudnn",
    strip_include_prefix = "cudnn/include",
    deps = [":cuda_headers"],
)

cc_import(
    name = "cufft",
    interface_library = "cuda/lib/%{cufft_lib}",
    system_provided = 1,
)

cc_import(
    name = "curand",
    interface_library = "cuda/lib/%{curand_lib}",
    system_provided = 1,
)

cc_library(
    name = "cuda",
    deps = [
        ":cublas",
        ":cublasLt",
        ":cuda_headers",
        ":cudart",
        ":cudnn",
        ":cufft",
        ":curand",
    ],
)

alias(
    name = "cub_headers",
    actual = "%{cub_actual}",
)

cuda_header_library(
    name = "cupti_headers",
    hdrs = [":cuda-extras"],
    include_prefix = "third_party/gpus",
    includes = ["cuda/extras/CUPTI/include/"],
    deps = [":cuda_headers"],
)

cuda_header_library(
    name = "nvml_headers",
    hdrs = [":nvml"],
    include_prefix = "third_party/gpus",
    includes = ["cuda/nvml/include/"],
    deps = [":cuda_headers"],
)

cc_import(
    name = "cupti_dsos",
    interface_library = "cuda/lib/%{cupti_lib}",
    system_provided = 1,
)

cc_import(
    name = "cusparse",
    interface_library = "cuda/lib/%{cusparse_lib}",
    system_provided = 1,
)

cc_library(
    name = "libdevice_root",
    data = [":cuda-nvvm"],
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
    srcs = ["cuda/cuda_config.py"],
)

%{copy_rules}
