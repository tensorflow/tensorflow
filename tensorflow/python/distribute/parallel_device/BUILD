load("//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

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
    srcs_version = "PY2AND3",
    deps = [
        ":parallel_device_ops",
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

tf_gen_op_wrapper_py(
    name = "parallel_device_ops_py",
    out = "gen_parallel_device_ops.py",
    deps = ["//tensorflow/c/eager/parallel_device:parallel_device_ops"],
)

tf_custom_op_library(
    name = "_parallel_device_ops.so",
    srcs = ["//tensorflow/c/eager/parallel_device:parallel_device_ops_srcs"],
)

tf_custom_op_py_library(
    name = "parallel_device_ops",
    dso = [":_parallel_device_ops.so"],
    kernels = ["//tensorflow/c/eager/parallel_device:parallel_device_ops"],
    visibility = ["//tensorflow:internal"],
    deps = [":parallel_device_ops_py"],
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
