load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_xla_ops",
    out = "gen_xla_ops.py",
    deps = [
        ":xla_ops",
    ],
)

tf_custom_op_library(
    name = "_xla_ops.so",
    srcs = [
        "xla_ops.cc",
    ],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "@com_google_absl//absl/algorithm:container",
    ],
)
