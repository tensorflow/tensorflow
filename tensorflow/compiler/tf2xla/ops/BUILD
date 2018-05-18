package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "xla_ops",
    srcs = [
        "dynamic_slice_ops.cc",
        "functional_ops.cc",
        "reduce_window_op.cc",
        "sendrecv_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
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
