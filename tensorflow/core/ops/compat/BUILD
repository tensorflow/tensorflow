# Description:
# For keeping the history of OpDefs for every major version of TensorFlow,
# to validate that we don't make backwards-incompatible changes.

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "op_compatibility_lib",
    srcs = ["op_compatibility_lib.cc"],
    hdrs = ["op_compatibility_lib.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "backwards_compatibility_test",
    size = "small",
    data = [
        ":ops_history.v0.pbtxt",
        "//tensorflow/core:ops/ops.pbtxt",
    ],
    deps = [
        ":op_compatibility_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_binary(
    name = "update_ops",
    srcs = ["update_ops_main.cc"],
    deps = [
        ":op_compatibility_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
