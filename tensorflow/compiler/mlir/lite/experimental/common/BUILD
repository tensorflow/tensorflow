load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

cc_library(
    name = "outline_operations",
    srcs = ["outline_operations.cc"],
    hdrs = ["outline_operations.h"],
    compatible_with = get_compatible_with_cloud(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow:cluster_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
