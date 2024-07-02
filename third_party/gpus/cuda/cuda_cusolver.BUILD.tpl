licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

cc_import(
    name = "cusolver",
    hdrs = [":headers"],
    shared_library = "lib/libcusolver.so.%{version}",
    visibility = ["//visibility:public"],
    deps = [
        "@cuda_nvjitlink//:nvjitlink",
        "@cuda_cusparse//:cusparse",
        "@cuda_cublas//:cublas",
        "@cuda_cublas//:cublasLt",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob([
        "include/**",
    ]),
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
