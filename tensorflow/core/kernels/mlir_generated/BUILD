# Generates CUDA kernels using MLIR codegen.

load("//tensorflow/core/kernels/mlir_generated:build_defs.bzl", "gen_kernel_library", "if_mlir_generated_gpu_kernels_enabled")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

package(
    default_visibility = ["//tensorflow/core/kernels:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "mlir_generated_gpu_kernels_enabled",
    define_values = {
        "tensorflow_enable_mlir_generated_gpu_kernels": "1",
    },
)

tf_kernel_library(
    name = "cwise_op",
    gpu_srcs = ["cwise_op_gpu_tanh.cu.cc"],
    tags = ["manual"],
    deps = if_cuda([
        ":tanh_kernels",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:stream_executor",
    ]),
)

tf_cuda_cc_test(
    name = "gpu_tanh_test",
    size = "small",
    srcs = if_mlir_generated_gpu_kernels_enabled(["gpu_tanh_test.cc"]),
    tags = tf_cuda_tests_tags() + ["no_rocm"],
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
