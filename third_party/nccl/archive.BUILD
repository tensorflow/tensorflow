# NVIDIA NCCL 2
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])

exports_files(["LICENSE.txt"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load(
    "@local_config_nccl//:build_defs.bzl",
    "cuda_rdc_library",
    "gen_device_srcs",
)

cc_library(
    name = "src_hdrs",
    hdrs = [
        "src/include/collectives.h",
        "src/nccl.h",
    ],
    strip_include_prefix = "src",
)

cc_library(
    name = "include_hdrs",
    hdrs = glob(["src/include/**"]),
    strip_include_prefix = "src/include",
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

cc_library(
    name = "device_hdrs",
    hdrs = glob(["src/collectives/device/*.h"]),
    strip_include_prefix = "src/collectives/device",
)

# NCCL compiles the same source files with different NCCL_OP/NCCL_TYPE defines.
# RDC compilation requires that each compiled module has a unique ID. Clang
# derives the module ID from the path only so we need to copy the files to get
# different IDs for different parts of compilation. NVCC does not have that
# problem because it generates IDs based on preprocessed content.
gen_device_srcs(
    name = "device_srcs",
    srcs = [
        "src/collectives/device/all_gather.cu.cc",
        "src/collectives/device/all_reduce.cu.cc",
        "src/collectives/device/broadcast.cu.cc",
        "src/collectives/device/reduce.cu.cc",
        "src/collectives/device/reduce_scatter.cu.cc",
        "src/collectives/device/sendrecv.cu.cc",
    ],
)

cuda_rdc_library(
    name = "device",
    srcs = [
        "src/collectives/device/functions.cu.cc",
        "src/collectives/device/onerank_reduce.cu.cc",
        ":device_srcs",
    ] + glob([
        # Required for header inclusion checking, see below for details.
        "src/collectives/device/*.h",
        "src/nccl.h",
    ]),
    deps = [
        ":device_hdrs",
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "net",
    srcs = [
        "src/transport/coll_net.cc",
        "src/transport/net.cc",
    ],
    linkopts = select({
        "@org_tensorflow//tensorflow:macos": [],
        "//conditions:default": ["-lrt"],
    }),
    deps = [
        ":include_hdrs",
        ":src_hdrs",
    ],
)

cc_library(
    name = "nccl",
    srcs = glob(
        include = [
            "src/**/*.cc",
            # Required for header inclusion checking, see below for details.
            "src/graph/*.h",
        ],
        # Exclude device-library code.
        exclude = [
            "src/collectives/device/**",
            "src/transport/coll_net.cc",
            "src/transport/net.cc",
        ],
    ) + [
        # Required for header inclusion checking (see
        # http://docs.bazel.build/versions/master/be/c-cpp.html#hdrs).
        # Files in src/ which #include "nccl.h" load it from there rather than
        # from the virtual includes directory.
        "src/include/collectives.h",
        "src/nccl.h",
    ],
    hdrs = ["src/nccl.h"],
    include_prefix = "third_party/nccl",
    linkopts = select({
        "@org_tensorflow//tensorflow:macos": [],
        "//conditions:default": ["-lrt"],
    }),
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
    deps = [
        ":device",
        ":include_hdrs",
        ":net",
        ":src_hdrs",
    ],
)
