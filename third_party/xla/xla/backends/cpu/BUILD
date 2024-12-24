load("//xla/backends/cpu:package_groups.bzl", "xla_cpu_backend_access")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

xla_cpu_backend_access()

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = ["alignment.h"],
    visibility = internal_visibility([":friends"]),
)

cc_library(
    name = "alignment",
    hdrs = ["alignment.h"],
    deps = ["@eigen_archive//:eigen3"],
)
