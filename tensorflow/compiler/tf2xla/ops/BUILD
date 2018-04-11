package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

cc_library(
    name = "functional_ops",
    srcs = ["functional_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "sendrecv_ops",
    srcs = ["sendrecv_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_functional_ops",
    out = "gen_functional_ops.py",
    deps = [
        ":functional_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_sendrecv_ops",
    out = "gen_sendrecv_ops.py",
    deps = [
        ":sendrecv_ops",
    ],
)
