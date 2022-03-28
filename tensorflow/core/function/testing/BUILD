load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(
    default_visibility = ["//tensorflow/core/function:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "test_pass_cc",
    testonly = 1,
    hdrs = [
        "test_pass.h",
    ],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_python_pybind_extension(
    name = "test_pass_pybind",
    testonly = 1,
    srcs = ["test_pass_pybind.cc"],
    module_name = "test_pass_pybind",
    deps = [
        ":test_pass_cc",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "test_pass_py",
    testonly = 1,
    srcs = [
        "test_pass.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":test_pass_pybind",
        "//tensorflow/python:pywrap_tensorflow",  # buildcleaner: keep (required for TF pybind)
    ],
)
