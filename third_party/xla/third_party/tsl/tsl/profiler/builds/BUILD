load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//tsl/profiler:internal"]),
    licenses = ["notice"],
)

# ONLY FOR DEV TESTING. DO NOT USE IF YOU DO NOT KNOW ABOUT IT ALREADY.
config_setting(
    name = "profiler_build_oss",
    define_values = {"profiler_build": "oss"},
)
