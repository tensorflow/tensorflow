package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_cc")

tf_gen_op_wrapper_cc(
    name = "functional_ops_gen",
    include_internal_ops = 1,
    out_ops_file = "ops/functional_ops",
    deps = ["//tensorflow/compiler/tf2xla/ops:functional_ops"],
)

cc_library(
    name = "functional_ops",
    srcs = ["ops/functional_ops.cc"],
    hdrs = ["ops/functional_ops.h"],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/ops:functional_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_wrapper_cc(
    name = "sendrecv_ops_gen",
    include_internal_ops = 1,
    out_ops_file = "ops/sendrecv_ops",
    deps = ["//tensorflow/compiler/tf2xla/ops:sendrecv_ops"],
)

cc_library(
    name = "sendrecv_ops",
    srcs = ["ops/sendrecv_ops.cc"],
    hdrs = ["ops/sendrecv_ops.h"],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/ops:sendrecv_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

# -----------------------------------------------------------------------------

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
