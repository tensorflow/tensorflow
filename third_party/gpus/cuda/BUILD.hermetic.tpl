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
            ":nvjitlink_headers",
            ":cusolver_headers",
            ":cufft_headers",
            ":cusparse_headers",
            ":curand_headers",
            ":cupti_headers",
            ":nvml_headers"],
)

cc_library(
    name = "cudart_static",
    srcs = ["@%{cudart_repo_name}//:static"],
    linkopts = [
        "-ldl",
        "-lpthread",
        %{cudart_static_linkopt}
    ],
)

alias(
  name = "cuda_driver",
  actual = "@%{cudart_repo_name}//:cuda_driver",
)

alias(
  name = "cudart_headers",
  actual = "@%{cudart_repo_name}//:headers",
)

alias(
  name = "cudart",
  actual = "@%{cudart_repo_name}//:cudart",
)

alias(
  name = "nvjitlink_headers",
  actual = "@%{nvjitlink_repo_name}//:headers",
)

alias(
  name = "nvjitlink",
  actual = "@%{nvjitlink_repo_name}//:nvjitlink",
)

alias(
  name = "nvtx_headers",
  actual = "@%{nvtx_repo_name}//:headers",
)

alias(
  name = "nvml_headers",
  actual = "@%{nvml_repo_name}//:headers",
)

alias(
  name = "nvcc_headers",
  actual = "@%{nvcc_repo_name}//:headers",
)

alias(
  name = "cccl_headers",
  actual = "@%{cccl_repo_name}//:headers",
)

alias(
  name = "cublas_headers",
  actual = "@%{cublas_repo_name}//:headers",
)

alias(
  name = "cusolver_headers",
  actual = "@%{cusolver_repo_name}//:headers",
)

alias(
  name = "cufft_headers",
  actual = "@%{cufft_repo_name}//:headers",
)

alias(
  name = "cusparse_headers",
  actual = "@%{cusparse_repo_name}//:headers",
)

alias(
  name = "curand_headers",
  actual = "@%{curand_repo_name}//:headers",
)

alias(
  name = "cublas",
  actual = "@%{cublas_repo_name}//:cublas",
)

alias(
  name = "cublasLt",
  actual = "@%{cublas_repo_name}//:cublasLt",
)

alias(
  name = "cusolver",
  actual = "@%{cusolver_repo_name}//:cusolver",
)

alias(
  name = "cudnn",
  actual = "@%{cudnn_repo_name}//:cudnn",
)

alias(
  name = "cudnn_ops_infer",
  actual = "@%{cudnn_repo_name}//:cudnn_ops_infer",
)

alias(
  name = "cudnn_cnn_infer",
  actual = "@%{cudnn_repo_name}//:cudnn_cnn_infer",
)

alias(
  name = "cudnn_ops_train",
  actual = "@%{cudnn_repo_name}//:cudnn_ops_train",
)

alias(
  name = "cudnn_cnn_train",
  actual = "@%{cudnn_repo_name}//:cudnn_cnn_train",
)

alias(
  name = "cudnn_adv_infer",
  actual = "@%{cudnn_repo_name}//:cudnn_adv_infer",
)

alias(
  name = "cudnn_adv_train",
  actual = "@%{cudnn_repo_name}//:cudnn_adv_train",
)
alias(
  name = "cudnn_header",
  actual = "@%{cudnn_repo_name}//:headers",
)

alias(
  name = "cufft",
  actual = "@%{cufft_repo_name}//:cufft",
)

alias(
  name = "curand",
  actual = "@%{curand_repo_name}//:curand",
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

alias(
  name = "cupti_headers",
  actual = "@%{cupti_repo_name}//:headers",
)

alias(
  name = "cupti_dsos",
  actual = "@%{cupti_repo_name}//:cupti",
)

alias(
  name = "cusparse",
  actual = "@%{cusparse_repo_name}//:cusparse",
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

py_library(
    name = "cuda_test_runner_py",
    srcs = ["cuda/cuda_test_runner.py"],
)

filegroup(name="cuda-include", srcs = ["@%{cudart_repo_name}//:include"])

filegroup(name="cccl-include", srcs = ["@%{cccl_repo_name}//:include"])

filegroup(name="nvtx-include", srcs = ["@%{nvtx_repo_name}//:include"])

filegroup(name="nvcc-include", srcs = ["@%{nvcc_repo_name}//:include"])

filegroup(name="cupti-include", srcs = ["@%{cupti_repo_name}//:include"])

filegroup(name="nvml-include", srcs = ["@%{nvml_repo_name}//:include"])

filegroup(name="nvjitlink-include", srcs = ["@%{nvjitlink_repo_name}//:include"])

filegroup(name="cuda-nvvm", srcs = ["@%{nvcc_repo_name}//:nvvm"])

filegroup(name="cublas-include", srcs = ["@%{cublas_repo_name}//:include"])

filegroup(name="cusolver-include", srcs = ["@%{cusolver_repo_name}//:include"])

filegroup(name="cufft-include", srcs = ["@%{cufft_repo_name}//:include"])

filegroup(name="cusparse-include", srcs = ["@%{cusparse_repo_name}//:include"])

filegroup(name="curand-include", srcs = ["@%{curand_repo_name}//:include"])

filegroup(name="cudnn-include", srcs = ["@%{cudnn_repo_name}//:include"])
