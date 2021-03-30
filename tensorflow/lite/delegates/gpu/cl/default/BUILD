package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "qcom_wrapper",
    hdrs = ["qcom_wrapper.h"],
)

cc_library(
    name = "recordable_queue",
    srcs = ["recordable_queue.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:recordable_queue",
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)
