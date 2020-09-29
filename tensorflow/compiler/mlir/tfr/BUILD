load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load(
    "//third_party/mlir:tblgen.bzl",
    "gentbl",
)
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//learning/brain/experimental/mlir/tfr/...",
        "//tensorflow/compiler/mlir/...",
    ],
)

filegroup(
    name = "tfr_ops_td_files",
    srcs = [
        "ir/tfr_ops.td",
        "//tensorflow/compiler/mlir/tensorflow:ir/tf_op_base.td",
        "//tensorflow/compiler/mlir/tensorflow:ir/tf_op_interfaces.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/Shape/IR/ShapeOps.td",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

gentbl(
    name = "tfr_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tfr_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tfr_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfr_ops.td",
    td_srcs = [
        ":tfr_ops_td_files",
    ],
)

cc_library(
    name = "tfr",
    srcs = [
        "ir/tfr_ops.cc",
        "ir/tfr_ops.cc.inc",
    ],
    hdrs = [
        "ir/tfr_ops.h",
        "ir/tfr_ops.h.inc",
        "ir/tfr_types.h",
    ],
    deps = [
        ":tfr_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils/utils.cc",
    ],
    hdrs = [
        "utils/utils.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "passes/canonicalize.cc",
        "passes/decompose.cc",
        "passes/raise_to_tf.cc",
    ],
    hdrs = [
        "passes/passes.h",
    ],
    deps = [
        ":tfr",
        ":utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFToStandard",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "tfr-opt",
    srcs = ["passes/tfr_opt.cc"],
    deps = [
        ":passes",
        ":tfr",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Shape",
        "@llvm-project//mlir:StandardOps",
    ],
)

glob_lit_tests(
    data = [
        ":test_utilities",
        "@llvm-project//mlir:run_lit.sh",
    ],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/tfr:tfr-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

cc_library(
    name = "tfr_decompose_ctx",
    srcs = ["integration/tfr_decompose_ctx.cc"],
    hdrs = ["integration/tfr_decompose_ctx.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/compiler/mlir/tfr",
        "//tensorflow/compiler/mlir/tfr:passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:optimization_registry",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "tfr_decompose_ctx_test",
    srcs = ["integration/tfr_decompose_ctx_test.cc"],
    deps = [
        ":tfr_decompose_ctx",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "graph_decompose_pass",
    srcs = ["integration/graph_decompose_pass.cc"],
    hdrs = ["integration/graph_decompose_pass.h"],
    data = ["//tensorflow/compiler/mlir/tfr/resources:decomposition_lib"],
    deps = [
        ":tfr_decompose_ctx",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)
