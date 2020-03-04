package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mlir",
    srcs = ["mlir.cc"],
    hdrs = ["mlir.h"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_utils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

filegroup(
    name = "pywrap_mlir_hdrs",
    srcs = [
        "mlir.h",
    ],
    visibility = [
        "//tensorflow/python:__pkg__",
    ],
)
