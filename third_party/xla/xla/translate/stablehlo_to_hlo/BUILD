load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

# Deprecated, use //third_party/tensorflow/compiler/xla/hlo/translate/stablehlo_to_hlo:translate #
# instead.
cc_library(
    name = "translate",
    hdrs = ["translate.h"],
    deps = [
        "//xla/hlo/translate/stablehlo_to_hlo:translate",
    ],
)
