load("//xla/tests:build_defs.bzl", "xla_test")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "custom_fusion",
    srcs = ["custom_fusion.cc"],
    hdrs = ["custom_fusion.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_kernel",
        "//xla:status",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "custom_fusion_pattern",
    srcs = ["custom_fusion_pattern.cc"],
    hdrs = ["custom_fusion_pattern.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "custom_kernel",
    srcs = ["custom_kernel.cc"],
    hdrs = ["custom_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:statusor",
        "//xla/stream_executor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# Bundle all custom fusions into a single target, so we can link all fusions and patterns by adding
# a single dependency.
cc_library(
    name = "custom_fusion_library",
    visibility = ["//visibility:public"],
    deps = [":cutlass_gemm_fusion"],
)

cc_library(
    name = "cutlass_gemm_fusion",
    srcs = ["cutlass_gemm_fusion.cc"],
    hdrs = ["cutlass_gemm_fusion.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_fusion",
        ":custom_fusion_pattern",
        ":custom_kernel",
        ":cutlass_gemm",
        ":cutlass_gemm_custom_kernel",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:pattern_matcher",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = 1,  # static fusion registration
)

xla_test(
    name = "cutlass_gemm_fusion_test",
    srcs = ["cutlass_gemm_fusion_test.cc"],
    backends = ["gpu"],
    deps = [
        ":custom_fusion_pattern",
        ":cutlass_gemm_fusion",
        "//xla:array",
        "//xla:array2d",
        "//xla:array3d",
        "//xla:debug_options_flags",
        "//xla:error_spec",
        "//xla:literal_util",
        "//xla:types",
        "//xla/service/gpu:custom_fusion_rewriter",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/tests:hlo_test_base",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# CUTLASS Gemm <-> xla::gpu::kernel::CustomKernel adaptor
#===--------------------------------------------------------------------------------------------===#

cc_library(
    name = "cutlass_gemm_custom_kernel",
    srcs = if_cuda_is_configured(
        ["cutlass_gemm_custom_kernel.cc"],
        ["cutlass_gemm_custom_kernel_stub.cc"],
    ),
    hdrs = ["cutlass_gemm_custom_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_kernel",
        ":cutlass_gemm",
        ":cutlass_gemm_kernels",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@cutlass_archive//:cutlass",
    ],
)

xla_test(
    name = "cutlass_gemm_custom_kernel_test",
    srcs = if_cuda_is_configured(["cutlass_gemm_custom_kernel_test.cc"]),
    backends = ["gpu"],
    deps = [
        ":cutlass_gemm_custom_kernel",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_platform",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_binary(
    name = "cutlass_gemm_custom_kernel_benchmarks",
    testonly = 1,
    srcs = if_cuda_is_configured(["cutlass_gemm_custom_kernel_benchmarks.cc"]),
    deps = [
        ":cutlass_gemm_custom_kernel",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/service:gpu_plugin",
        "//xla/stream_executor",
        "//xla/stream_executor:multi_platform_manager",
        "//xla/stream_executor:platform",
        "//xla/stream_executor/cuda:cuda_platform",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
    ],
)

#===--------------------------------------------------------------------------------------------===#
# CUTLASS GemmUniversal-base kernels <-> StreamExecutor adaptor
#===--------------------------------------------------------------------------------------------===#

cc_library(
    name = "cutlass_gemm",
    hdrs = ["cutlass_gemm.h"],
    visibility = ["//visibility:public"],
)

cuda_library(
    name = "cutlass_gemm_adaptor",
    hdrs = if_cuda_is_configured(["cutlass_gemm_adaptor.cu.h"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cutlass_gemm",
        "@cutlass_archive//:cutlass",
    ]),
)

#===--------------------------------------------------------------------------------------------===#
# CUTLASS Gemm kernels implementation
#===--------------------------------------------------------------------------------------------===#

# We split each individual kernel into a separate targets to compile them all in parallel. We also
# do not have any dependencies except CUTLASS itself to reduce the number of recompilations.

cc_library(
    name = "cutlass_gemm_kernels",
    visibility = ["//visibility:public"],
    deps = [
        ":cutlass_gemm_kernel_bf16xbf16_to_bf16",
        ":cutlass_gemm_kernel_bf16xbf16_to_bf16_sm80",
        ":cutlass_gemm_kernel_f32xf32_to_f32",
    ],
)

# CUTLASS requires all loops to be unrolled, and in some kernels defined below we force Clang/LLVM
# to unroll them with extra compiler options because by default LLVM is not as aggressive with loop
# unrolling as NVCC.

# TODO(ezhulenev): Write a build rule to simplify kernel target declarations.

cuda_library(
    name = "cutlass_gemm_kernel_bf16xbf16_to_bf16",
    srcs = if_cuda_is_configured(["cutlass_gemm_kernel_bf16xbf16_to_bf16.cu.cc"]),
    copts = ["-mllvm -unroll-threshold=100000"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cutlass_gemm_adaptor",
        "@cutlass_archive//:cutlass",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cuda_library(
    name = "cutlass_gemm_kernel_bf16xbf16_to_bf16_sm80",
    srcs = if_cuda_is_configured(["cutlass_gemm_kernel_bf16xbf16_to_bf16_sm80.cu.cc"]),
    copts = ["-mllvm -unroll-threshold=100000"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cutlass_gemm_adaptor",
        "@cutlass_archive//:cutlass",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cuda_library(
    name = "cutlass_gemm_kernel_f32xf32_to_f32",
    srcs = if_cuda_is_configured(["cutlass_gemm_kernel_f32xf32_to_f32.cu.cc"]),
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured([
        ":cutlass_gemm_adaptor",
        "@cutlass_archive//:cutlass",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)
