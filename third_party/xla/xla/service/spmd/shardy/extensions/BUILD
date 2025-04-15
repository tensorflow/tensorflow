load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/deepmind/partir/compiler/mpmd/...",
        "//learning/deepmind/partir/compiler/shardonnay/...",
        "//third_party/openxla/shardy/tools/...",
        "//xla/...",
    ],
)

cc_library(
    name = "mhlo_extensions",
    srcs = ["mhlo_extensions.cc"],
    hdrs = ["mhlo_extensions.h"],
    deps = [
        "//xla/mlir_hlo",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@shardy//shardy/dialect/sdy/transforms/propagation:op_sharding_rule_builder",
    ],
)
