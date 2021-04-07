# TensorFlow -> TOSA Compiler Bridge.
# See:
#   https://developer.mlplatform.org/w/tosa/
#   https://github.com/llvm/llvm-project/blob/master/mlir/docs/Dialects/TOSA.md

load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//third_party/mlir:tblgen.bzl", "gentbl")

# TODO: Tighten visibility once targets are at the right granularity.
package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/...",
    ],
)

package_group(
    name = "friends",
    includes = [
        ":internal",
    ],
    packages = [
        "//third_party/iree/...",
    ],
)

filegroup(
    name = "tosa_ops_td_files",
    srcs = [
        "@llvm-project//mlir:TosaDialectTdFiles",
    ],
    compatible_with = get_compatible_with_cloud(),
)

gentbl(
    name = "tosa_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-pass-decls -name LegalizeTosa",
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes_header",
    hdrs = [
        "transforms/passes.h",
        "transforms/passes.h.inc",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = ["@llvm-project//mlir:Pass"],
)

cc_library(
    name = "legalize_common",
    srcs = [
        "transforms/legalize_common.cc",
        "transforms/legalize_utils.cc",
    ],
    hdrs = [
        "transforms/legalize_common.h",
        "transforms/legalize_utils.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
    ],
    alwayslink = 1,
)

gentbl(
    name = "tosa_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/tf_legalize_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_legalize_patterns.td",
    td_srcs = [
        ":tosa_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tf_passes",
    srcs = [
        "tf_passes.cc",
        "transforms/fuse_bias_tf.cc",
        "transforms/legalize_tf.cc",
        "transforms/tf_legalize_patterns.inc",
    ],
    hdrs = [
        "tf_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    visibility = [":friends"],
    deps = [
        ":legalize_common",
        ":passes_header",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "tosa_legalize_tfl_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/tfl_legalize_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tfl_legalize_patterns.td",
    td_srcs = [
        ":tosa_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
    ],
)

cc_library(
    name = "tfl_passes",
    srcs = [
        "tfl_passes.cc",
        "transforms/convert_tfl_uint8.cc",
        "transforms/legalize_tfl.cc",
        "transforms/tfl_legalize_patterns.inc",
    ],
    hdrs = [
        "tfl_passes.h",
        "transforms/passes.h",
    ],
    compatible_with = get_compatible_with_cloud(),
    visibility = [":friends"],
    deps = [
        ":legalize_common",
        ":passes_header",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TosaDialect",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)
