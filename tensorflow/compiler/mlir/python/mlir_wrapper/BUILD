load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")
load("//rules_pywrap:pywrap.bzl", "pybind_extension", "pywrap_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "mlir_wrapper_lib",
    srcs = [
        "attrs.cc",
        "basic_classes.cc",
        "builders.cc",
        "ops.cc",
        "types.cc",
    ],
    hdrs = ["mlir_wrapper.h",],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

pybind_extension(
    name = "mlir_wrapper",
    srcs = [
        "mlir_wrapper.cc",
    ],
    data = [
        "mlir_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":mlir_wrapper_lib",
    ],
)

pybind_extension(
    name = "filecheck_wrapper",
    srcs = ["filecheck_wrapper.cc"],
    data = [
        "filecheck_wrapper.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "@llvm-project//llvm:FileCheckLib",
        "@llvm-project//llvm:Support",
    ],
)
