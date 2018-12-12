# NVIDIA NCCL 2
# A package of optimized primitives for collective multi-GPU communication.

licenses(["restricted"])

exports_files(["LICENSE.txt"])

load(
    "@local_config_nccl//:build_defs.bzl",
    "gen_nccl_h",
    "nccl_library",
    "rdc_copts",
    "rdc_library",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
)

# Generate the nccl.h header file.
gen_nccl_h(
    name = "nccl_h",
    output = "src/nccl.h",
    template = "src/nccl.h.in",
)

nccl_library(
    name = "src_hdrs",
    hdrs = [
        "src/nccl.h",
        # src/include/common_coll.h #includes "collectives/collectives.h".
        # All other #includes of collectives.h are patched in process_srcs.
        "src/collectives/collectives.h",
    ],
    strip_include_prefix = "src",
)

nccl_library(
    name = "include_hdrs",
    hdrs = glob(["src/include/*.h"]),
    strip_include_prefix = "src/include",
)

filegroup(
    name = "device_hdrs",
    srcs = glob(["src/collectives/device/*.h"]),
)

filegroup(
    name = "device_srcs",
    srcs = [
        "src/collectives/device/all_gather.cu",
        "src/collectives/device/all_reduce.cu",
        "src/collectives/device/broadcast.cu",
        "src/collectives/device/reduce.cu",
        "src/collectives/device/reduce_scatter.cu",
    ],
)

nccl_library(
    name = "sum",
    srcs = [
        ":device_hdrs",
        ":device_srcs",
    ],
    copts = ["-DNCCL_OP=0"] + rdc_copts(),
    linkstatic = True,
    prefix = "sum_",
    deps = [
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

nccl_library(
    name = "prod",
    srcs = [
        ":device_hdrs",
        ":device_srcs",
    ],
    copts = ["-DNCCL_OP=1"] + rdc_copts(),
    linkstatic = True,
    prefix = "_prod",
    deps = [
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

nccl_library(
    name = "min",
    srcs = [
        ":device_hdrs",
        ":device_srcs",
    ],
    copts = ["-DNCCL_OP=2"] + rdc_copts(),
    linkstatic = True,
    prefix = "min_",
    deps = [
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

nccl_library(
    name = "max",
    srcs = [
        ":device_hdrs",
        ":device_srcs",
    ],
    copts = ["-DNCCL_OP=3"] + rdc_copts(),
    linkstatic = True,
    prefix = "max_",
    deps = [
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

nccl_library(
    name = "functions",
    srcs = [
        "src/collectives/device/functions.cu",
        ":device_hdrs",
    ],
    copts = rdc_copts(),
    linkstatic = True,
    deps = [
        ":include_hdrs",
        ":src_hdrs",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

rdc_library(
    name = "device_code",
    deps = [
        ":functions",
        ":max",
        ":min",
        ":prod",
        ":sum",
    ],
)

# Primary NCCL target.
nccl_library(
    name = "nccl",
    srcs = glob(
        include = ["src/**/*.cu"],
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
    copts = cuda_default_copts(),
    include_prefix = "third_party/nccl",
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
    deps = [
        ":device_code",
        ":include_hdrs",
        ":src_hdrs",
    ],
)
