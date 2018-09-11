# AMD rccl
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_default_copts", "if_rocm")

SRCS = [
    "src/rccl.cpp",
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
    name = "rccl",
    srcs = if_rocm(SRCS_CU_CC + glob(["src/*.h"])),
    hdrs = if_rocm(["inc/rccl/rccl.h"]),
    copts = [
        "-Iexternal/rccl_archive/src",
        "-Iexternal/rccl_archive/inc",
        "-O3",
    ] + rocm_default_copts(),
    linkopts = select({
        "//conditions:default": [
            "-lrt",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = ["@local_config_rocm//rocm:rocm_headers"],
)
