load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_gpu_library",
)

tf_gpu_library(
    name = "cupti_wrapper",
    srcs = [
        "cupti_wrapper.cc",
    ],
    hdrs = [
        "cupti_wrapper.h",
    ],
    copts = tf_copts(),
    gpu_deps = [
        "//tensorflow/core:stream_executor",
        "@local_config_cuda//cuda:cupti_headers",
    ],
    data = ["@local_config_cuda//cuda:cupti_dsos"],
    visibility = ["//visibility:public"],
)
