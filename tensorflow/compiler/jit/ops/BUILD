load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    visibility = [
        "//tensorflow/compiler/tf2xla:internal",
        "//tensorflow/python:__pkg__",
    ],
    deps = ["//tensorflow/compiler/jit/ops:xla_ops"],
)

py_strict_library(
    name = "xla_ops_grad",
    srcs = ["xla_ops_grad.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/framework:ops"],
)
