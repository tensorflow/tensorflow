load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)

tf_cuda_library(
    name = "cupti_wrapper",
    srcs = [
        "cupti_wrapper.cc",
    ],
    hdrs = [
        "cupti_wrapper.h",
    ],
    copts = tf_copts(),
    cuda_deps = [
        "//tensorflow/core:stream_executor",
        "//third_party/gpus/cuda:cuda_headers",
        "//third_party/gpus/cuda:cupti_headers",
    ],
    data = ["//third_party/gpus/cuda:cupti_dsos"],
    visibility = ["//visibility:public"],
)
