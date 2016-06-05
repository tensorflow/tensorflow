load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "roi_pooling_op.so",
    srcs = ["roi_pooling_op.cc"],
    gpu_srcs = ["roi_pooling_op.cu.cc"],
) 

tf_custom_op_library(
    name = "roi_pooling_op_grad.so",
    srcs = ["roi_pooling_op_grad.cc"],
    gpu_srcs = ["roi_pooling_op.cu.cc"],
)
