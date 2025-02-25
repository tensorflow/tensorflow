# Description:
#   Contains User Ops (internal TensorFlow version).

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")

visibility = [
    "//tensorflow:internal",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
    out = "ops/gen_user_ops.py",
    py_lib_rule = py_strict_library,
)

# This target is deprecated.
py_strict_library(
    name = "ops",
    srcs = ["user_ops.py"],
    deps = [
        ":user_ops_gen",
        "//tensorflow/python/util:tf_export",
    ],
)
