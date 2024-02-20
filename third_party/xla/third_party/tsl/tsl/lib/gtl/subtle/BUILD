# Description:
# gtl subtle packages.

load("//tsl:tsl.bzl", "internal_visibility")
load("//tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "map_traits",
    srcs = [
        "map_traits.h",
    ],
    visibility = internal_visibility([
        "//tensorflow/core/lib/gtl/subtle:__pkg__",
        "//tsl/lib/gtl:__pkg__",
    ]),
)
