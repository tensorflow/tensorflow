# NVIDIA NCCL 2
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])

exports_files(["LICENSE.txt"])

load(
    "@local_config_nccl//:build_defs.bzl",
    "cuda_rdc_library",
    "gen_device_srcs",
    "process_srcs",
)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cuda_library")

process_srcs(
    name = "process_srcs",
    srcs = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "src_hdrs",
    hdrs = [
        "src/collectives/collectives.h",
        "src/nccl.h",
    ],
    data = [":process_srcs"],
    strip_include_prefix = "src",
)

cc_library(
    name = "include_hdrs",
    hdrs = glob(["src/include/*.h"]),
    data = [":process_srcs"],
    strip_include_prefix = "src/include",
)

cc_library(
    name = "device_hdrs",
    hdrs = glob(["src/collectives/device/*.h"]),
    strip_include_prefix = "src/collectives/device",
)

filegroup(
    name = "device_srcs",
    srcs = [
        "src/collectives/device/all_gather.cu.cc",
        "src/collectives/device/all_reduce.cu.cc",
        "src/collectives/device/broadcast.cu.cc",
        "src/collectives/device/reduce.cu.cc",
        "src/collectives/device/reduce_scatter.cu.cc",
    ],
)

# NCCL compiles the same source files with different NCCL_OP defines. RDC
# compilation requires that each compiled module has a unique ID. Clang derives
# the module ID from the path only so we need to rename the files to get
# different IDs for different parts of compilation. NVCC does not have that
# problem because it generates IDs based on preprocessed content.
gen_device_srcs(
    name = "sum",
    srcs = [":device_srcs"],
    NCCL_OP = 0,
)

gen_device_srcs(
    name = "prod",
    srcs = [":device_srcs"],
    NCCL_OP = 1,
)

gen_device_srcs(
    name = "min",
    srcs = [":device_srcs"],
    NCCL_OP = 2,
)

gen_device_srcs(
    name = "max",
    srcs = [":device_srcs"],
    NCCL_OP = 3,
)

cuda_rdc_library(
    name = "device",
    srcs = [
        "src/collectives/device/functions.cu.cc",
        ":max",
        ":min",
        ":prod",
        ":sum",
    ],
    deps = [
        ":device_hdrs",
        ":include_hdrs",
        ":src_hdrs",
    ],
)

# Primary NCCL target.
tf_cuda_library(
    name = "nccl",
    srcs = glob(
        include = ["src/**/*.cu.cc"],
        # Exclude device-library code.
        exclude = ["src/collectives/device/**"],
    ) + [
        # Required for header inclusion checking (see
        # http://docs.bazel.build/versions/master/be/c-cpp.html#hdrs).
        # Files in src/ which #include "nccl.h" load it from there rather than
        # from the virtual includes directory.
        "src/nccl.h",
    ],
    hdrs = ["src/nccl.h"],
    copts = ["-Wno-vla"],
    include_prefix = "third_party/nccl",
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
    deps = [
        ":device",
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cudart_static",
    ],
)
