# Generates headers containing cubin for CUDA kernels.
load("//tensorflow/core/kernels/cubin_headers:build_defs.bzl", "gen_kernel_library")

package(
    default_visibility = ["//tensorflow/core/kernels:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

gen_kernel_library(
    name = "bias_add",
    same_shape = "0,2",
    tile_size = "16x16",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

gen_kernel_library(
    name = "relu",
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

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
