load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "roi_pooling_op.so",
    srcs = ["roi_pooling_op.cc"],
) 

tf_custom_op_library(
    name = "roi_pooling_op_gpu.so",
    srcs = ["roi_pooling_op_gpu.cc"],
    gpu_srcs = ["roi_pooling_op_gpu.cu.cc"],
)

tf_custom_op_library(
    name = "cuda_op_kernel.so",
    srcs = ["cuda_op_kernel.cc"],
    gpu_srcs = ["cuda_op_kernel.cu.cc"],
)