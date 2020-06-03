# Generates headers containing cubin for CUDA kernels.
load("//tensorflow/core/kernels/cubin_headers:build_defs.bzl", "gen_kernel_library")

package(
    default_visibility = ["//tensorflow/core/kernels:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

bias_add_kernel = """
func @bias_add(%arg0: tensor<?x?xf99>,
         %arg1: tensor<?xf99>) -> tensor<?x?xf99> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) { T = "tfdtype$DT_TYPE" }
    : (tensor<?x?xf99>, tensor<?xf99>) -> tensor<?x?xf99>
  return %0 : tensor<?x?xf99>
}
"""

gen_kernel_library(
    name = "bias_add",
    op = bias_add_kernel,
    same_shape = "0,2",
    tile_size = "16x16",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

relu_kernel = """
func @relu(%arg0: tensor<?xf99>) -> tensor<?xf99> {
  %0 = "tf.Relu"(%arg0) { T = "tfdtype$DT_TYPE" }
    : (tensor<?xf99>) -> tensor<?xf99>
  return %0 : tensor<?xf99>
}
"""

gen_kernel_library(
    name = "relu",
    op = relu_kernel,
    same_shape = "0,1",
    tile_size = "256",
    types = [
        "f16",
        "f32",
        "f64",
    ],
)

tanh_kernel = """
func @tanh(%arg0: tensor<?xf99>) -> tensor<?xf99> {
  %0 = "tf.Tanh"(%arg0) { T = "tfdtype$DT_TYPE" }
    : (tensor<?xf99>) -> tensor<?xf99>
  return %0 : tensor<?xf99>
}
"""

gen_kernel_library(
    name = "tanh",
    op = tanh_kernel,
    tile_size = "256",
    types = [
        "f32",
        "f64",
    ],
)
