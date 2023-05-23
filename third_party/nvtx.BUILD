#Description : NVIDIA Tools Extension (NVTX) library for adding profiling annotations to applications.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["restricted"])  # NVIDIA proprietary license

filegroup(
    name = "nvtx_header_files",
    srcs = glob([
        "nvtx3/**",
    ]),
)

cc_library(
    name = "nvtx",
    hdrs = [":nvtx_header_files"],
    include_prefix = "third_party",
)
