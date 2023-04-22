load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    default_visibility = ["//tensorflow/compiler/tf2xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "xla_ops_wrapper_py",
    out = "xla_ops.py",
    deps = ["//tensorflow/compiler/jit/ops:xla_ops"],
)

py_library(
    name = "xla_ops_grad",
    srcs = ["xla_ops_grad.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python:framework_ops"],
)
