load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "support",
    srcs = ["support.cc"],
    hdrs = ["support.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:decision",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "support_test",
    srcs = ["support_test.cc"],
    deps = [
        ":support",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/rocm:rocm_compute_capability",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "hlo_to_tensor_ir",
    srcs = ["hlo_to_tensor_ir.cc"],
    hdrs = ["hlo_to_tensor_ir.h"],
    deps = [
        ":layout_utils",
        "//xla/backends/gpu/codegen/tensor_ir/transforms:passes",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_function_importer",
        "//xla/mlir/utils:error_util",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@stablehlo//:stablehlo_ops",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)

cc_library(
    name = "compilation_pipeline",
    srcs = ["compilation_pipeline.cc"],
    hdrs = ["compilation_pipeline.h"],
    # tileiras, ptxas and libnvvm staged from the CUDA 13.3+ toolkit;
    # FindCudaExecutable("tileiras") picks them up from runfiles.
    data = if_google(["@local_config_cuda//cuda:runtime_tileiras"]),
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tensor_ir//:NVTensorIRConversionOptions",
        "@tensor_ir//:NVTensorIRCudaTilePipelines",
        "@tensor_ir//:NVTensorIRTileIRAssembly",
        "@tsl//tsl/platform:path",
    ],
)

cc_library(
    name = "fusion",
    srcs = ["fusion.cc"],
    hdrs = ["fusion.h"],
    # The tensor_ir compiler and runtime libraries include cuda.h; builds
    # without CUDA link :fusion_stub instead.
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    deps = [
        ":compilation_pipeline",
        ":custom_kernel",
        ":hlo_to_tensor_ir",
        ":layout_utils",
        ":support",
        "//xla:future",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/mlir/utils:error_util",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@tensor_ir//:NVTensorIRCompiler",
        "@tensor_ir//:NVTensorIRCompilerOptions",
        "@tensor_ir//:NVTensorIRConversionOptions",
        "@tensor_ir//:NVTensorIRDialect",
        "@tensor_ir//:NVTensorIRUtils",
    ],
)

cc_library(
    name = "layout_utils",
    srcs = ["layout_utils.cc"],
    hdrs = ["layout_utils.h"],
    deps = [
        "//xla:shape_util",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)

cc_library(
    name = "custom_kernel",
    srcs = ["custom_kernel.cc"],
    hdrs = ["custom_kernel.h"],
    # The tensor_ir runtime library includes cuda.h.
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@tensor_ir//:NVTensorIRRuntime",
    ],
)

cc_library(
    name = "fusion_stub",
    srcs = ["fusion_stub.cc"],
    hdrs = ["fusion.h"],
    visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:ir_emitter_context",
        "@com_google_absl//absl/status",
    ],
)

xla_cc_test(
    name = "layout_utils_test",
    srcs = ["layout_utils_test.cc"],
    deps = [
        ":layout_utils",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tensor_ir//:NVTensorIRDialect",
        "@tensor_ir//:NVTensorIRSupport",
    ],
)

xla_cc_test(
    name = "hlo_to_tensor_ir_test",
    srcs = ["hlo_to_tensor_ir_test.cc"],
    deps = [
        ":hlo_to_tensor_ir",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)
