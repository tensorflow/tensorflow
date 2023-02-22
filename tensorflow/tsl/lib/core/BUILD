# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

# TODO(rdzhabarov): Tighten visibility after migration is complete.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "bitmap.h",
        "bits.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "bitmap.h",
        "bits.h",
        "status_test_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_core_all_tests",
    srcs = [
        "bitmap_test.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_core_headers",
    srcs = [
        "bitmap.h",
        "bits.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_core_status_test_util_header",
    srcs = [
        "status_test_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/core:__pkg__",
    ],
)

cc_library(
    name = "bitmap",
    srcs = ["bitmap.cc"],
    hdrs = ["bitmap.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "status_test_util",
    testonly = 1,
    hdrs = ["status_test_util.h"],
    deps = [
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "bits",
    hdrs = ["bits.h"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
    ],
)
