# Description:
#   Contains kernels for the input pipeline.

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

cc_library(
    name = "input_pipeline_kernels",
    srcs = ["input_pipeline_kernels.cc"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)
