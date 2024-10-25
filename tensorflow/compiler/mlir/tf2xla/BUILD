# Description:
#    TF2XLA Bridge and related components.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "tensorflow_mlir_tf2xla",
    packages = [
        "//tensorflow/compiler/mlir/tf2xla/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

alias(
    name = "compile_mlir_util",
    actual = "//tensorflow/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
)
