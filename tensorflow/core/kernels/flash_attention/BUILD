# Description:
#   FlashAttention and FlashAttentionGrad op kernels: fused scaled dot-product
#   attention that never materializes the [seq_q, seq_kv] attention matrix.
#
#   On sm80+ GPUs, float16/bfloat16 inputs with matching query/value head dims
#   that are a multiple of 8 (up to 256) use the FlashAttention-2 kernels
#   (@flash_attention). Everything else uses the op's own memory-efficient CUDA
#   or CPU kernels.

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

# FlashAttention-2 only supports sm80+. With clang, skip compiling its kernels
# for older archs altogether. Otherwise they compile to stubs, and
# flash_attention_fa2::Supported() keeps older GPUs off this path at run time.
_FA2_PRE_SM80_ARCHS = [
    "sm_50",
    "sm_52",
    "sm_60",
    "sm_61",
    "sm_70",
    "sm_75",
]

# Stand-ins for the few PyTorch headers the FlashAttention-2 sources include.
cc_library(
    name = "torch_shim",
    hdrs = glob(["torch_shim/**"]),
    includes = ["torch_shim"],
    visibility = ["//visibility:private"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

# The FlashAttention-2 kernels and a TF-independent C++ API for them. Where
# @flash_attention isn't available, fa2_api.cu.cc builds a stub that reports
# FA2 as unsupported, so the op uses its own memory-efficient kernels.
cuda_library(
    name = "fa2",
    srcs = ["fa2_api.cu.cc"] + if_oss(["@flash_attention//:sm80_kernel_srcs"]),
    hdrs = ["fa2_api.h"],
    copts = [
        "-fvisibility=hidden",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-missing-braces",
    ] + if_oss(select({
        "//tensorflow:is_cuda_clang": [
            # Overrides the toolchain's "--maxrregcount=0", which makes ptxas
            # clamp kernels without launch bounds to too few registers for FA2.
            "-Xcuda-ptxas",
            "--maxrregcount=255",
            # ptxas -O3 on the backward kernels can exceed the CI's 1h action
            # timeout.
            "-Xcuda-ptxas",
            "-O2",
        ] + ["--no-cuda-gpu-arch=" + arch for arch in _FA2_PRE_SM80_ARCHS],
        "//conditions:default": [],
    })),
    local_defines = [
        "FLASH_NAMESPACE=flash_attention_fa2_impl",
        "FLASHATTENTION_DISABLE_ALIBI",
        # fa2_api.cu.cc never uses sliding-window attention.
        "FLASHATTENTION_DISABLE_LOCAL",
        "FLASHATTENTION_DISABLE_SOFTCAP",
    ],
    deps = [
        ":torch_shim",
        "@local_config_cuda//cuda:cuda_headers",
    ] + if_oss([
        "@cutlass_archive//:cutlass",
        "@flash_attention//:headers",
        "@local_config_cuda//cuda:cudart_headers",
    ]),
)

# Device functors shared by the TF kernels and the XLA custom calls.
cc_library(
    name = "flash_attention_functor",
    hdrs = ["flash_attention_op.h"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "flash_attention_op",
    srcs = ["flash_attention_op.cc"],
    gpu_deps = [":fa2"],
    gpu_srcs = [
        "flash_attention_op_gpu.cu.h",
        "flash_attention_op_fwd_gpu.cu.cc",
        "flash_attention_op_bwd_gpu.cu.cc",
    ],
    deps = [
        ":flash_attention_functor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status",
        "@eigen_archive//:eigen3",
    ],
)

# XLA custom-call targets that run the GPU kernels inside XLA clusters.
cc_library(
    name = "flash_attention_custom_call",
    srcs = if_cuda(["flash_attention_custom_call.cc"]),
    hdrs = ["flash_attention_custom_call.h"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    visibility = ["//visibility:private"],
    deps = if_cuda([
        ":flash_attention_functor",
        ":flash_attention_op_gpu",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/service:custom_call_target_registry",
    ]),
    alwayslink = 1,
)

# Lowers FlashAttention/FlashAttentionGrad to the custom calls above when
# compiling for GPU with XLA. Linked into //tensorflow/compiler/tf2xla/kernels.
cc_library(
    name = "flash_attention_xla_op",
    srcs = if_cuda(["flash_attention_xla_op.cc"]),
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    deps = if_cuda([
        ":fa2",
        ":flash_attention_custom_call",
        ":flash_attention_functor",
        ":flash_attention_op_gpu",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
    ]),
    alwayslink = 1,
)
