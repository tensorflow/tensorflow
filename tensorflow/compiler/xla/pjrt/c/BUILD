load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)
