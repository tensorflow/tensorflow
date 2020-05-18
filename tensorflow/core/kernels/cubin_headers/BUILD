# Generates headers containing cubin for CUDA kernels.
load("//tensorflow/core/kernels/cubin_headers:build_defs.bzl", "gen_kernel_image_hdr")

bias_add_kernel = """
func @bias_add(%arg0: tensor<?x?xf99>,
         %arg1: tensor<?xf99>) -> tensor<?x?xf99> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) { T = "tfdtype$DT_TYPE" }
    : (tensor<?x?xf99>, tensor<?xf99>) -> tensor<?x?xf99>
  return %0 : tensor<?x?xf99>
}
"""

[
    gen_kernel_image_hdr(
        name = "bias_add_{type}_kernel".format(type = type),
        op = bias_add_kernel.replace("f99", type).replace("DT_TYPE", dtype),
        same_shape = "0,2",
        tile_size = "16x16",
    )
    for (type, dtype) in [
        ("f16", "DT_HALF"),
        ("f32", "DT_FLOAT"),
        ("f64", "DT_DOUBLE"),
    ]
]

relu_kernel = """
func @relu(%arg0: tensor<?xf99>) -> tensor<?xf99> {
  %0 = "tf.Relu"(%arg0) { T = "tfdtype$DT_TYPE" }
    : (tensor<?xf99>) -> tensor<?xf99>
  return %0 : tensor<?xf99>
}
"""

[
    gen_kernel_image_hdr(
        name = "relu_{type}_kernel".format(type = type),
        op = relu_kernel.replace("f99", type).replace("DT_TYPE", dtype),
        same_shape = "0,1",
        tile_size = "256",
    )
    for (type, dtype) in [
        ("f16", "DT_HALF"),
        ("f32", "DT_FLOAT"),
        ("f64", "DT_DOUBLE"),
    ]
]
