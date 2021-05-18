package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "custom_parsers",
    srcs = ["custom_parsers.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_parsers.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:operation_parser",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "custom_transformations",
    srcs = ["custom_transformations.cc"],
    hdrs = ["//tensorflow/lite/delegates/gpu/common:custom_transformations.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
    ],
)
