load("//tensorflow:tensorflow.bzl", "py_strict_library")

package(
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

py_strict_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)
