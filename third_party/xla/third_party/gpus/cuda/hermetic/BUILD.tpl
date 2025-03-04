load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

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
cc_library(
    name = "cuda_headers",
    hdrs = [
        "cuda/cuda_config.h",
    ],
    include_prefix = "third_party/gpus",
    includes = [
        ".",  # required to include cuda/cuda/cuda_config.h as cuda/config.h
    ],
    deps = [":cudart_headers",
            ":cublas_headers",
            ":cccl_headers",
            ":nvtx_headers",
            ":nvcc_headers",
            ":cusolver_headers",
            ":cufft_headers",
            ":cusparse_headers",
            ":curand_headers",
            ":cupti_headers",
            ":nvml_headers",
            ":nvjitlink_headers"],
)

# This target is needed by the `cuda_library` rule. We can't implicitly
# depend on `:cuda_headers` directly since the user may explicit depend
# on `:cuda_headers` and duplicated dependencies are not allowed in Bazel.
# There is also no good way to deduplicate dependencies, but an alias works
# just fine.
alias(
    name = "implicit_cuda_headers_dependency",
    actual = ":cuda_headers",
)

cc_library(
    name = "cudart_static",
    srcs = ["@cuda_cudart//:static"],
    linkopts = [
        "-ldl",
        "-lpthread",
        %{cudart_static_linkopt}
    ],
)

alias(
  name = "cuda_runtime",
  actual = ":cudart_static",
)

alias(
    name = "cuda_driver",
    actual = select({
        "@cuda_driver//:forward_compatibility": "@cuda_driver//:nvidia_driver",
        "//conditions:default": "@cuda_cudart//:cuda_driver",
    }),
)

alias(
  name = "cudart_headers",
  actual = "@cuda_cudart//:headers",
)

alias(
  name = "cudart",
  actual = "@cuda_cudart//:cudart",
)

alias(
  name = "nvtx_headers",
  actual = "@cuda_nvtx//:headers",
)

alias(
  name = "nvml_headers",
  actual = "@cuda_nvml//:headers",
)

alias(
  name = "nvcc_headers",
  actual = "@cuda_nvcc//:headers",
)

alias(
  name = "cccl_headers",
  actual = "@cuda_cccl//:headers",
)

alias(
  name = "cublas_headers",
  actual = "@cuda_cublas//:headers",
)

alias(
  name = "cusolver_headers",
  actual = "@cuda_cusolver//:headers",
)

alias(
  name = "cufft_headers",
  actual = "@cuda_cufft//:headers",
)

alias(
  name = "cusparse_headers",
  actual = "@cuda_cusparse//:headers",
)

alias(
  name = "curand_headers",
  actual = "@cuda_curand//:headers",
)

alias(
  name = "nvjitlink_headers",
  actual = "@cuda_nvjitlink//:headers",
)

alias(
  name = "cublas",
  actual = "@cuda_cublas//:cublas",
)

alias(
  name = "cublasLt",
  actual = "@cuda_cublas//:cublasLt",
)

alias(
  name = "cusolver",
  actual = "@cuda_cusolver//:cusolver",
)

alias(
  name = "cudnn",
  actual = "@cuda_cudnn//:cudnn",
)

alias(
  name = "cudnn_header",
  actual = "@cuda_cudnn//:headers",
)

alias(
  name = "cufft",
  actual = "@cuda_cufft//:cufft",
)

alias(
  name = "curand",
  actual = "@cuda_curand//:curand",
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
    actual = ":cuda_headers",
)

alias(
  name = "cupti_headers",
  actual = "@cuda_cupti//:headers",
)

alias(
  name = "cupti_dsos",
  actual = "@cuda_cupti//:cupti",
)

alias(
  name = "cusparse",
  actual = "@cuda_cusparse//:cusparse",
)

alias(
    name = "cuda-nvvm",
    actual = "@cuda_nvcc//:nvvm",
)

alias(
    name = "nvjitlink",
    actual = "@cuda_nvjitlink//:nvjitlink"
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

# Config setting whether TensorFlow is built with CUDA.
alias(
    name = "cuda_tools",
    actual = "@local_config_cuda//:is_cuda_enabled",
)

# Flag indicating if we should include CUDA libs.
bool_flag(
    name = "include_cuda_libs",
    build_setting_default = False,
)

config_setting(
    name = "cuda_libs",
    flag_values = {":include_cuda_libs": "True"},
)

# This flag should be used only when someone wants to build the wheel with CUDA
# dependencies.
bool_flag(
    name = "override_include_cuda_libs",
    build_setting_default = False,
)

config_setting(
    name = "overrided_cuda_libs",
    flag_values = {":override_include_cuda_libs": "True"},
)

selects.config_setting_group(
    name = "any_cuda_libs",
    match_any = [
        ":cuda_libs",
        ":overrided_cuda_libs"
    ],
)

selects.config_setting_group(
    name = "cuda_tools_and_libs",
    match_all = [
        ":any_cuda_libs",
        ":cuda_tools"
    ],
)

cc_library(
    # This is not yet fully supported, but we need the rule
    # to make bazel query happy.
    name = "nvptxcompiler",
)
