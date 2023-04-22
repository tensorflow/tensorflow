load("//tensorflow:tensorflow.bzl", "pytype_library")  # buildifier: disable=same-origin-load

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_library(
    name = "ops",
    srcs = ["tpu_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:tpu_ops_gen",
        "//tensorflow/python/tpu:tpu_function",
    ],
)
