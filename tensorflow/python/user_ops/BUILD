# Description:
#   Contains User Ops (internal TensorFlow version).

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
)

# This target is deprecated.
py_library(
    name = "ops",
    srcs = ["user_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":user_ops_gen",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)
