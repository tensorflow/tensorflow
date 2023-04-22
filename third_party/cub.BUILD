# Description: CUB library which is a set of primitives for GPU programming.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

cc_library(
    name = "cub",
    hdrs = glob(["cub/**"]),
    deps = ["@local_cuda//:cuda_headers"],
)
