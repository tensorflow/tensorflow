# Forwarding targets for libraries moved to //xla/backends/gpu/cost_model.
#
# TODO: Remove once all external users (e.g. XProf OSS, which pins an XLA
# commit) have migrated to //xla/backends/gpu/cost_model.
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    hdrs = ["gpu_hlo_cost_analysis.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//xla/backends/gpu/cost_model:gpu_hlo_cost_analysis",
    ],
)
