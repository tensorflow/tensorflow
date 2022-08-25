# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "byte_order",
    hdrs = ["byte_order.h"],
    compatible_with = get_compatible_with_portable(),
)

cc_library(
    name = "gif",
    hdrs = ["gif.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@gif",
    ],
)

cc_library(
    name = "stack_frame",
    hdrs = ["stack_frame.h"],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "lib_hdrs",
    srcs = [
        "stack_frame.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Header files for tensorflow/core:platform_base.
filegroup(
    name = "base_hdrs",
    srcs = [
        "byte_order.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "framework_lite_hdrs",
    srcs = [
        "byte_order.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "byte_order.h",
    ],
    compatible_with = get_compatible_with_portable(),
)

filegroup(
    name = "gif_hdrs",
    srcs = [
        "gif.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/core/lib/gif:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_internal_headers",
    srcs = glob(
        [
            "*.h",
        ],
        exclude = [
            "gif.h",
        ],
    ),
    compatible_with = get_compatible_with_portable(),
)
