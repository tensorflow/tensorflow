# Description: CUB library which is a set of primitives for GPU programming.
load("@local_cuda//:defs.bzl", "if_local_cuda")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

cc_library(
    name = "cub",
    hdrs = glob(["cub/**"]),
    deps = if_local_cuda(["@local_cuda//:cuda_headers"]),
)
