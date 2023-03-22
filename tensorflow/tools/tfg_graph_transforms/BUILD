# Description:
#   Utilities to perform MLIR TFG graph transformations.

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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

TFG_GRAPH_TRANSFORM_DEPS = [
    ":utils",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:Transforms",
    "//tensorflow/compiler/mlir:init_mlir",
    "//tensorflow/compiler/mlir/tensorflow",
    "//tensorflow/compiler/mlir/tensorflow:error_util",
    "//tensorflow/core:lib",
    "//tensorflow/core/ir:Dialect",
    "//tensorflow/core/protobuf:for_core_protos_cc",
    "//tensorflow/core/transforms:PassRegistration",
    "//tensorflow/compiler/tf2xla/ops:xla_ops",
    "//tensorflow/core:ops",
    "//tensorflow/core/ir/importexport:graphdef_export",
    "//tensorflow/core/ir/importexport:graphdef_import",
    "//tensorflow/core/ir/importexport:savedmodel_export",
    "//tensorflow/core/ir/importexport:savedmodel_import",
    "//tensorflow/core/ir:tf_op_registry",
]

# Description:
#   A tool that provides a mechanism to run TFG graph optimizations operating on the
#   GraphDef from the supplied SavedModel as an input.
tf_cc_binary(
    name = "tfg_graph_transforms",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    deps = TFG_GRAPH_TRANSFORM_DEPS,
)

# Description:
#    The tool wrapped into a library, so that it could be used
#    when custom defined ops and passes need to be added.
cc_library(
    name = "tfg_graph_transforms_main",
    srcs = [
        "tfg_graph_transforms_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = TFG_GRAPH_TRANSFORM_DEPS,
)
