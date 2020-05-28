package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "default_selector",
    srcs = ["default_selector.cc"],
    deps = [
        "//tensorflow/lite/delegates/gpu/cl:model_hints",
        "//tensorflow/lite/delegates/gpu/cl:tensor_type",
        "//tensorflow/lite/delegates/gpu/cl/kernels:gpu_operation",
        "//tensorflow/lite/delegates/gpu/cl/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)
