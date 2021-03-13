load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(licenses = ["notice"])

tf_python_pybind_extension(
    name = "mlir_wrapper",
    srcs = [
        "attrs.cc",
        "basic_classes.cc",
        "builders.cc",
        "mlir_wrapper.cc",
        "mlir_wrapper.h",
        "ops.cc",
        "types.cc",
    ],
    module_name = "mlir_wrapper",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "filecheck_wrapper",
    srcs = ["filecheck_wrapper.cc"],
    module_name = "filecheck_wrapper",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@pybind11",
    ],
)
