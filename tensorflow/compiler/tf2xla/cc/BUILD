package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_cc")

tf_gen_op_wrapper_cc(
    name = "xla_ops_gen",
    out_ops_file = "ops/xla_ops",
    deps = ["//tensorflow/compiler/tf2xla/ops:xla_ops"],
)

cc_library(
    name = "xla_ops",
    srcs = ["ops/xla_ops.cc"],
    hdrs = ["ops/xla_ops.h"],
    copts = ["-g", "-O0", ],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_gen_op_wrapper_cc(
    name = "xla_jit_op_gen",
    out_ops_file = "ops/xla_jit_op",
    deps = ["//tensorflow/compiler/jit/ops:xla_ops"],
)

cc_library(
    name = "xla_jit_ops",
    srcs = ["ops/xla_jit_op.cc"],
    hdrs = ["ops/xla_jit_op.h"],
    copts = ["-g", "-O0", ],
    deps = [
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)
