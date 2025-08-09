licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "cublas_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcublas.so.%{libcublas_version}",
    deps = [":cublasLt"],
)

cc_import(
    name = "cublasLt_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcublasLt.so.%{libcublaslt_version}",
)
%{multiline_comment}
cc_library(
    name = "cublas",
    visibility = ["//visibility:public"],
    %{comment}deps = [":cublas_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cublas/lib"),
)

cc_library(
    name = "cublasLt",
    visibility = ["//visibility:public"],
    %{comment}deps = [":cublasLt_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cublas/lib"),
)

cc_library(
    name = "headers",
    %{comment}hdrs = [
        %{comment}"include/cublas.h",
        %{comment}"include/cublasLt.h",
        %{comment}"include/cublas_api.h",
        %{comment}"include/cublas_v2.h",
    %{comment}],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
