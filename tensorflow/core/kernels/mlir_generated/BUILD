# Generates CUDA kernels using MLIR codegen.

load(
    "//tensorflow/core/kernels/mlir_generated:build_defs.bzl",
    "gen_kernel_library",
    "if_mlir_generated_gpu_kernels_enabled",
)
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/tools/kernel_gen:__subpackages__",
        "//tensorflow/core/kernels:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "mlir_generated_gpu_kernels_disabled",
    define_values = {
        "tensorflow_enable_mlir_generated_gpu_kernels": "0",
    },
)

tf_kernel_library(
    name = "cwise_unary_op",
    # Technically these source files don't need --config=cuda or --config=rocm,
    # but we want to avoid building them if they are not needed.
    srcs = if_cuda_or_rocm([
        "cwise_op_gpu_abs.cc",
        "cwise_op_gpu_base.cc",
        "cwise_op_gpu_base.h",
        "cwise_op_gpu_tanh.cc",
    ]),
    # Building abs_kernels and tanh_kernels requires hexdump on the ROCM
    # platform, which might not be available.
    tags = ["no_rocm"],
    deps = if_cuda_or_rocm([
        ":abs_kernels",
        ":tanh_kernels",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
    ]),
)

tf_cuda_cc_test(
    name = "gpu_tanh_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_tanh_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

tf_cuda_cc_test(
    name = "gpu_abs_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_abs_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

# TODO(b/160731748): Re-enable when it works again.
# gen_kernel_library(
#     name = "bias_add",
#     same_shape = "0,2",
#     tile_size = "16x16",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

# TODO(b/160190568): Re-enable when it works again.
# gen_kernel_library(
#     name = "relu",
#     same_shape = "0,1",
#     tile_size = "256",
#     types = [
#         "f16",
#         "f32",
#         "f64",
#     ],
# )

gen_kernel_library(
    name = "tanh",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "abs",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
        "i32",
        "i64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "ceil",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "cos",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "exp",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "floor",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "log",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "neg",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "rsqrt",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)

gen_kernel_library(
    name = "sqrt",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
    unroll_factors = "4",
)
