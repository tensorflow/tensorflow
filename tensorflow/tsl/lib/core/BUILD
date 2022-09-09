# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# TODO(rdzhabarov): Tighten visibility after migration is complete.
package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "bitmap.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_core_all_headers",
    srcs = [
        "bitmap.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
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
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//tensorflow/core:__pkg__"],
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
