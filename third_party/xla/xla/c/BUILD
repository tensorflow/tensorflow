load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/tfrt/tpu_plugin:__subpackages__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "c_api_decl",
    hdrs = [
        "c_api_decl.h",
    ],
    deps = [
    ],
)
