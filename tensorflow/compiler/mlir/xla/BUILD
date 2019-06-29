load("@local_config_mlir//:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//babelfish/device/...",
        "//learning/brain/experimental/mlir/...",
        "//tensorflow/compiler/mlir/...",
        "//third_party/mlir_edge/...",
    ],
)

filegroup(
    name = "xla_ops_td_files",
    srcs = [
        "ir/xla_ops.td",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "xla_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/xla_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/xla_ops.cc.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/xla_ops.td",
    td_srcs = [
        ":xla_ops_td_files",
    ],
)

gentbl(
    name = "xla_legalize_tf_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_legalize_tf.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    td_srcs = [
        ":xla_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
    ],
    deps = [
        ":xla",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "xla_legalize_to_standard_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_legalize_to_standard.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/legalize_to_standard_patterns.td",
    td_srcs = [
        ":xla_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_legalize_control_flow",
    srcs = [
        "transforms/legalize_control_flow.cc",
    ],
    deps = [
        ":xla",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_standard",
    srcs = [
        "transforms/legalize_to_standard.cc",
    ],
    deps = [
        ":xla",
        ":xla_legalize_to_standard_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla",
    srcs = [
        "ir/xla_ops.cc",
        "ir/xla_ops.cc.inc",
        "ir/xla_ops.h.inc",
    ],
    hdrs = [
        "ir/xla_ops.h",
        "transforms/passes.h",
    ],
    includes = ["include"],
    deps = [
        ":xla_ops_inc_gen",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:TypeUtilities",
    ],
    alwayslink = 1,
)

# Library with XLA dialect static initialization.
cc_library(
    name = "xla_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    deps = [
        ":xla",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)
