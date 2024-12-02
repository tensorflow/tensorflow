load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "collectives",
    hdrs = ["collectives.h"],
    deps = [
        ":communicator",
    ],
)

cc_library(
    name = "communicator",
    hdrs = ["communicator.h"],
)
