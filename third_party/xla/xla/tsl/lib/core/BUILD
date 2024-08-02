# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

# TODO(rdzhabarov): Tighten visibility after migration is complete.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "legacy_lib_core_status_test_util_header",
    srcs = [
        "status_test_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//tensorflow/core:__pkg__",
        "@local_tsl//tsl/lib/core:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
    ]),
)

cc_library(
    name = "status_test_util",
    testonly = 1,
    hdrs = ["status_test_util.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test",
    ],
)
