# Description:
# CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance
# matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE.txt"])

filegroup(
    name = "cutlass_header_files",
    srcs = glob([
        "include/**",
    ]),
)

cc_library(
    name = "cutlass",
    hdrs = [":cutlass_header_files"],
    strip_include_prefix = "/include",
)
