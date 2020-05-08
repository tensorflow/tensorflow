load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(licenses = ["notice"])

tf_python_pybind_extension(
    name = "mlir_wrapper",
    srcs = ["mlir_wrapper.cc"],
    module_name = "mlir_wrapper",
    visibility = ["//visibility:public"],
    deps = [
        ":mlir_util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "@llvm-project//mlir:StandardOps",
        "@pybind11",
    ],
)

cc_library(
    name = "mlir_util",
    srcs = [
        "attrs.cc",
        "basic_classes.cc",
        "builders.cc",
        "ops.cc",
        "types.cc",
    ],
    hdrs = ["mlir_util.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@pybind11",
    ],
)
