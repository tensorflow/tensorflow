#Description : NVIDIA Tools Extension (NVTX) library for adding profiling annotations to applications.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # NVIDIA proprietary license

#exports_files(["LICENSE.TXT"])

filegroup(
    name = "nvtx_header_files",
    srcs = glob([
        #"usr/local/cuda-*/targets/x86_64-linux/include/**",
        "nvtx3/**",
    ]),
)

cc_library(
    name = "nvtx",
    #hdrs = if_cuda([":nvtx_header_files"]),
    hdrs = [":nvtx_header_files"],
    include_prefix = "third_party",
    deps = [
        #"@local_config_cuda//cuda:cuda_headers",
    ],
)
