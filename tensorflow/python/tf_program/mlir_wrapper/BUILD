load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(licenses = ["notice"])

tf_python_pybind_extension(
    name = "mlir_wrapper",
    srcs = ["mlir_wrapper.cc"],
    module_name = "mlir_wrapper",
    deps = [
        ":mlir_util",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "@pybind11",
        "@llvm-project//mlir:StandardOps",
        "//tensorflow/compiler/mlir/tensorflow",
    ],
    visibility = ["//visibility:public"]
)
cc_library(
    name = "mlir_util",
    hdrs = ["mlir_util.h"],
    srcs = ["basic_classes.cc", "types.cc", "builders.cc", "ops.cc", "attrs.cc"],
    deps = [
        "@pybind11",
        "@llvm-project//mlir:IR",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:StandardOps",
    ]
)
