package(
    default_testonly = 1,
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "all_generators",
    hdrs = [
        "add.h",
    ],
    deps = [
        ":add",
        "//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common/testing/feature_parity:utils",
        "//tensorflow/lite/kernels:builtin_ops",
        "@flatbuffers",
    ],
)
