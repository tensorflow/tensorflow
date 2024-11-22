load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "function_library",
    hdrs = ["function_library.h"],
    deps = [
        "//xla:util",
        "//xla/stream_executor/host:host_kernel_c_api",
        "@com_google_absl//absl/status:statusor",
    ],
)
