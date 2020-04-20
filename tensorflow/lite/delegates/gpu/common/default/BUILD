package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)
