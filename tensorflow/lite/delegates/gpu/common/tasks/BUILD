package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)
