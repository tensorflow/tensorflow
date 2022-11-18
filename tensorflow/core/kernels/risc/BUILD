load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "risc",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/kernels/risc/experimental",
    ],
)
