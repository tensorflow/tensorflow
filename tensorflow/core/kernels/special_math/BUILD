load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

# Implementation of Special Functions kernels.

package(
    default_visibility = ["//visibility:public"],
    features = [
        "parse_headers",
        "-layering_check",
    ],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "special_math_op",
    prefix = "special_math_op",
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_grad",
        "//tensorflow/core/framework:bounds_check",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:fill_functor",
        "//third_party/eigen3",
    ],
)
