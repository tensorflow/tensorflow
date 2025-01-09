# Description:
# gtl subtle packages.

load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "map_traits",
    srcs = [
        "map_traits.h",
        "@local_xla//xla/tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//tensorflow/core/lib/gtl:__pkg__"],
)
