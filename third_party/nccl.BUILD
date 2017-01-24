# NVIDIA nccl
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])  # BSD

exports_files(["LICENSE.txt"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "if_cuda")

SRCS = [
    "src/all_gather.cu",
    "src/all_reduce.cu",
    "src/broadcast.cu",
    "src/core.cu",
    "src/libwrap.cu",
    "src/reduce.cu",
    "src/reduce_scatter.cu",
]

# Copy .cu to .cu.cc so they can be in srcs of cc_library.
[
    genrule(
        name = "gen_" + src,
        srcs = [src],
        outs = [src + ".cc"],
        cmd = "cp $(location " + src + ") $(location " + src + ".cc)",
    )
    for src in SRCS
]

SRCS_CU_CC = [src + ".cc" for src in SRCS]

cc_library(
    name = "nccl",
    srcs = if_cuda(SRCS_CU_CC + glob(["src/*.h"])),
    hdrs = if_cuda(["src/nccl.h"]),
    copts = [
        "-DCUDA_MAJOR=0",
        "-DCUDA_MINOR=0",
        "-DNCCL_MAJOR=0",
        "-DNCCL_MINOR=0",
        "-DNCCL_PATCH=0",
        "-Iexternal/nccl_archive/src",
        "-O3",
    ] + cuda_default_copts(),
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)
