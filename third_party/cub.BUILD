# Description: CUB library which is a set of primitives for GPU programming.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "if_cuda")

filegroup(
    name = "cub_header_files",
    srcs = glob([
        "cub/**",
    ]),
)

cc_library(
    name = "cub",
    hdrs = if_cuda([":cub_header_files"]),
    include_prefix = "third_party",
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
