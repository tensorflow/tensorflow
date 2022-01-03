# Description:
#   Utilities to perform MLIR TFG graph transformations.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

cc_library(
    name = "import",
    srcs = ["import.cc"],
    hdrs = ["import.h"],
    deps = [
        ":utils",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/compiler/tf2xla/ops:xla_ops",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:ops",
        "//tensorflow/core:portable_jpeg_internal",
        "//tensorflow/core/ir/importexport:import",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "export",
    srcs = ["export.cc"],
    hdrs = ["export.h"],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/importexport:export",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

# Description:
#   A tool that provides a mechanism to run TFG graph optimizations operating on the
#   GraphDef from the supplied SavedModel as an input.
tf_cc_binary(
    name = "tfg_graph_transforms",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    deps = [
        ":export",
        ":import",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
