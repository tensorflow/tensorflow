# Description:
# Contains script to generate tensorflow/python/platform/build_info.py
load("@rules_python//python:defs.bzl", "py_binary")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "gen_build_info",
    srcs = ["gen_build_info.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no-remote-exec"],
    deps = [
        "@local_config_cuda//cuda:cuda_config_py",
        "@local_config_tensorrt//:tensorrt_config_py",
        "@six_archive//:six",
    ],
)
