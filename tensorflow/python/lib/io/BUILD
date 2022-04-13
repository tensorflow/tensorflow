# python/lib/io package

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_file_io",
    srcs = ["file_io_wrapper.cc"],
    module_name = "_pywrap_file_io",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_library(
    name = "lib",
    srcs = [
        "file_io.py",
        "python_io.py",
        "tf_record.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_file_io",
        ":_pywrap_record_io",
        "//tensorflow/python:errors",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_record_io",
    srcs = ["record_io_wrapper.cc"],
    module_name = "_pywrap_record_io",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/platform:types",
        "//tensorflow/python/lib/core:pybind11_absl",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/memory",
        "@pybind11",
    ],
)

tf_py_test(
    name = "file_io_test",
    size = "small",
    srcs = ["file_io_test.py"],
    python_version = "PY3",
    tags = [
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":lib",
        "//tensorflow/python:errors",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_test(
    name = "tf_record_test",
    size = "small",
    srcs = ["tf_record_test.py"],
    python_version = "PY3",
    deps = [
        ":lib",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/platform:client_testlib",
    ],
)
