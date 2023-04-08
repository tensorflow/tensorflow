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

cc_library(
    name = "mlir_bridge_rollout_policy",
    srcs = ["mlir_bridge_rollout_policy.cc"],
    hdrs = ["mlir_bridge_rollout_policy.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//mlir:IR",
    ],
)

alias(
    name = "compile_mlir_util",
    actual = "//tensorflow/compiler/mlir/tf2xla/api/v0:compile_mlir_util_no_tf_dialect_passes",
)

# Targets in the transform package we alias here to avoid updating all clients.
aliased_targets = [
    "legalize_tf",
    "legalize_utils",
    "tf_xla_passes",
    "xla_legalize_tf",
    "xla_legalize_tf_no_fallback",
    "xla_legalize_tf_with_tf2xla",
]

[
    alias(
        name = target,
        actual = "//tensorflow/compiler/mlir/tf2xla/transforms:%s" % target,
    )
    for target in aliased_targets
]
