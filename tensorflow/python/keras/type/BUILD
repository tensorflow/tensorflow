load("//tensorflow:tensorflow.bzl", "py_strict_library")

package(
    default_visibility = ["//tensorflow/python/keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_strict_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
)
