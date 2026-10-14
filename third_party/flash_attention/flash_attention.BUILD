# Description:
# FlashAttention-2 (https://github.com/Dao-AILab/flash-attention): fused,
# memory-efficient attention kernels for NVIDIA sm80+ GPUs, built on CUTLASS/CuTe.
#
# Only the kernel sources and headers are exported. They are compiled by
# //tensorflow/core/kernels/flash_attention:fa2, together with a small shim for
# the few PyTorch headers they include.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD 3-clause

exports_files(["LICENSE"])

# Head dimensions to instantiate kernels for. Must match the dispatch in
# //tensorflow/core/kernels/flash_attention:fa2_api.cu.cc.
_HEAD_DIMS = [32, 64, 96, 128, 192, 256]

cc_library(
    name = "headers",
    hdrs = glob(["src/*.h"]),
    includes = ["src"],
    deps = ["@cutlass_archive//:cutlass"],
)

# Forward and backward sm80 kernels for float16/bfloat16, with and without a
# causal mask. These run on any sm80+ GPU.
filegroup(
    name = "sm80_kernel_srcs",
    srcs = [
        "src/flash_{}_hdim{}_{}{}_sm80.cu".format(direction, head_dim, dtype, causal)
        for direction in ["fwd", "bwd"]
        for head_dim in _HEAD_DIMS
        for dtype in ["fp16", "bf16"]
        for causal in ["", "_causal"]
    ],
)
