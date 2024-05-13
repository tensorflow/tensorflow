load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")

visibility = ["//tensorflow/core/function:__subpackages__"]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
)

licenses(["notice"])

cc_library(
    name = "test_pass_cc",
    testonly = 1,
    hdrs = [
        "test_pass.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:logging",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_python_pybind_extension(
    name = "test_pass_pybind",
    testonly = 1,
    srcs = ["test_pass_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "test_pass_pybind.pyi",
    ],
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
    visibility = visibility,
    deps = [
        ":test_pass_pybind",
        "//tensorflow/python:pywrap_tensorflow",  # buildcleaner: keep (required for TF pybind)
    ],
)
