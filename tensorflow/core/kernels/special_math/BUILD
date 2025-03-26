load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")

# Implementation of Special Functions kernels.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "special_math_op",
    gpu_deps = ["//tensorflow/core/kernels:cwise_lib"],
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
        "@eigen_archive//:eigen3",
    ],
)
