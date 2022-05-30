package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
    ],
)
