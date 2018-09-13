package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/algorithm:container",
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
