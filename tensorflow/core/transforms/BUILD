load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//tensorflow:tensorflow.bzl", "tf_native_cc_binary")
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")

package(
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        ":__subpackages__",
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "--name",
                "TFGraph",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "PassDetail",
    hdrs = ["pass_detail.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "PassRegistration",
    hdrs = ["pass_registration.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/transforms/const_dedupe_hoist:Pass",
        "//tensorflow/core/transforms/drop_unregistered_attribute:OutputShapesPass",
        "//tensorflow/core/transforms/graph_to_func:Pass",
        "//tensorflow/core/transforms/toposort:Pass",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
tf_native_cc_binary(
    name = "tfg-transforms-opt",
    srcs = ["tfg-transforms-opt.cc"],
    deps = [
        ":PassRegistration",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TransformUtils",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-transforms-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
