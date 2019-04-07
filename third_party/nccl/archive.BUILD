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
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

cc_library(
    name = "src_hdrs",
    hdrs = process_srcs([
        "src/collectives/collectives.h",
        "src/nccl.h.in",
    ]),
)

cc_library(
    name = "include_hdrs",
    hdrs = process_srcs(glob(["src/include/*.h"])),
    strip_include_prefix = "include",
)

device_srcs = process_srcs([
    "src/collectives/device/all_gather.cu",
    "src/collectives/device/all_reduce.cu",
    "src/collectives/device/broadcast.cu",
    "src/collectives/device/reduce.cu",
    "src/collectives/device/reduce_scatter.cu",
])

# NCCL compiles the same source files with different NCCL_OP defines. RDC
# compilation requires that each compiled module has a unique ID. Clang derives
# the module ID from the path only so we need to rename the files to get
# different IDs for different parts of compilation. NVCC does not have that
# problem because it generates IDs based on preprocessed content.
gen_device_srcs(
    name = "sum",
    srcs = device_srcs,
    NCCL_OP = 0,
)

gen_device_srcs(
    name = "prod",
    srcs = device_srcs,
    NCCL_OP = 1,
)

gen_device_srcs(
    name = "min",
    srcs = device_srcs,
    NCCL_OP = 2,
)

gen_device_srcs(
    name = "max",
    srcs = device_srcs,
    NCCL_OP = 3,
)

cuda_rdc_library(
    name = "device",
    srcs = [
        ":max",
        ":min",
        ":prod",
        ":sum",
    ] + process_srcs(glob([
        "src/collectives/device/*.h",
        "src/collectives/device/functions.cu",
    ])),
    deps = [
        ":include_hdrs",
        ":src_hdrs",
    ],
)

# Primary NCCL target.
cc_library(
    name = "nccl",
    srcs = process_srcs(glob(
        include = ["src/**/*.cu"],
        # Exclude device-library code.
        exclude = ["src/collectives/device/**"],
    )) + [
        # Required for header inclusion checking (see
        # http://docs.bazel.build/versions/master/be/c-cpp.html#hdrs).
        "nccl.h",
        "collectives/collectives.h",
    ],
    hdrs = ["nccl.h"],
    copts = cuda_default_copts() + ["-Wno-vla"],
    include_prefix = "third_party/nccl",
    visibility = ["//visibility:public"],
    deps = [
        ":device",
        ":include_hdrs",
        "@local_config_cuda//cuda:cudart_static",
    ],
)
