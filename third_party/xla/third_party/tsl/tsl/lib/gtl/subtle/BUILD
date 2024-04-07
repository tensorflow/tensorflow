# Description:
# gtl subtle packages.

load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

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
