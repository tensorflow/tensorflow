load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "dlpack",
    srcs = ["dlpack.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:pywrap_tensorflow",
    ],
)

cuda_py_test(
    name = "dlpack_test",
    srcs = ["dlpack_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dlpack",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
