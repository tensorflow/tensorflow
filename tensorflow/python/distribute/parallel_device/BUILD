package(
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
    srcs_version = "PY2AND3",
    deps = [
        ":saving",
        "//tensorflow/python:_pywrap_parallel_device",
    ],
)

py_library(
    name = "saving",
    srcs = ["saving.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:framework_ops"],
)

py_test(
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
