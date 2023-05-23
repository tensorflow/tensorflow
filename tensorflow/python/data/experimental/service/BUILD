load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_server_lib",
    srcs = ["server_lib_wrapper.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/data/service:server_lib_headers_lib",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "//third_party/python_runtime:headers",
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)

py_strict_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    srcs_version = "PY3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":_pywrap_server_lib",
        ":_pywrap_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.py"],
    deps = [
        ":server_lib",
        "//tensorflow/python:errors",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/profiler:profiler_client",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_utils",
    srcs = ["utils_wrapper.cc"],
    deps = [
        "//tensorflow/core/data/service:py_utils",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "service",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":server_lib",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
    ],
)
