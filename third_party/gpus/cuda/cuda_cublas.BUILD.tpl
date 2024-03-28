licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "include",
    srcs = ["include/cublas.h", "include/cublas_v2.h", "include/cublas_api.h", "include/cublasLt.h"],
)

cc_import(
    name = "cublas",
    hdrs = [":headers"],
    shared_library = "lib/libcublas.so.%{version}",
    visibility = ["//visibility:public"],
)

cc_import(
    name = "cublasLt",
    hdrs = [":headers"],
    shared_library = "lib/libcublasLt.so.%{version}",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cublas_lib",
    srcs = ["lib/libcublas.so.%{version}"],
)

filegroup(
    name = "cublasLt_lib",
    srcs = ["lib/libcublasLt.so.%{version}"],
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
