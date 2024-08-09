licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])
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
