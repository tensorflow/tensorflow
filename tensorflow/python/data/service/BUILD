load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_python_pybind_extension(
    name = "_pywrap_server_lib",
    srcs = ["server_lib_wrapper.cc"],
    module_name = "_pywrap_server_lib",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/data/service:server_lib_headers_lib",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@pybind11",
    ],
)

py_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":_pywrap_server_lib",
    ],
)

tf_py_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.py"],
    deps = [
        ":server_lib",
        "//tensorflow/python:platform_test",
    ],
)
