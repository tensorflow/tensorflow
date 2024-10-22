load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

alias(
    name = "xla-translate",
    actual = "//xla/hlo/translate:xla-translate",
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate:xla-translate instead.",
)

alias(
    name = "xla-translate-opt",
    actual = "//xla/hlo/translate:xla-translate-opt",
    deprecation = "This library is deprecated. Use //third_party/tensorflow/compiler/xla/hlo/translate:xla-translate-opt instead.",
)
