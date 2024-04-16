# NVIDIA NVTX 3

licenses(["notice"])

exports_files(["LICENSE.txt"])

cc_library(
    name = "headers",
    hdrs = glob(["**"]),
    include_prefix = "nvtx3",
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
