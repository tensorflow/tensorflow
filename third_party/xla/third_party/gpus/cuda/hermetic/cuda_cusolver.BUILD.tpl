licenses(["restricted"])  # NVIDIA proprietary license
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "cuda_rpath_flags",
)

%{multiline_comment}
cc_import(
    name = "cusolver_shared_library",
    hdrs = [":headers"],
    shared_library = "lib/libcusolver.so.%{libcusolver_version}",
    deps = [
        "@cuda_nvjitlink//:nvjitlink",
        "@cuda_cusparse//:cusparse",
        "@cuda_cublas//:cublas",
        "@cuda_cublas//:cublasLt",
    ],
)
%{multiline_comment}
cc_library(
    name = "cusolver",
    %{comment}deps = [":cusolver_shared_library"],
    %{comment}linkopts = cuda_rpath_flags("nvidia/cusolver/lib"),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "headers",
    %{comment}hdrs = glob([
        %{comment}"include/cusolver*.h",
    %{comment}]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
