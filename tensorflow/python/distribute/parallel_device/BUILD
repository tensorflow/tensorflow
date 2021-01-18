load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

# Pybind rules must live in tensorflow/python due to header rule visibility.
exports_files(
    ["pywrap_parallel_device.cc"],
    visibility = ["//tensorflow/python:__pkg__"],
)

py_library(
    name = "parallel_device",
    srcs = ["parallel_device.py"],
    srcs_version = "PY3",
    deps = [
        ":saving",
        "//tensorflow/python:_pywrap_parallel_device",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/tpu:tpu_ops",
    ],
)

py_library(
    name = "saving",
    srcs = ["saving.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python:framework_ops"],
)

distribute_py_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.py"],
    python_version = "PY3",
    tags = [
        # Dependencies aren't otherwise included in the pip package yet.
        "no_pip",
    ],
    deps = [
        ":parallel_device",
        ":saving",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/module",
        "//tensorflow/python/tpu",
    ],
)
