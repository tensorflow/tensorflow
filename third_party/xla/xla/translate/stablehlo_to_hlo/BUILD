load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "translate",
    hdrs = ["translate.h"],
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate/stablehlo_to_hlo:translate instead.",
    deps = [
        "//xla/hlo/translate/stablehlo_to_hlo:translate",
    ],
)
