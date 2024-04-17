load("@local_xla//xla/tsl:tsl.bzl", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//tensorflow:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility([
        "//platforms/performance/tf_sim/utils:__subpackages__",
        "//platforms/xla/service:__subpackages__",
        "//tensorflow:__subpackages__",
    ]),
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tsl_cc_test(
    name = "math_util_test",
    size = "small",
    srcs = [
        "math_util_test.cc",
    ],
    deps = [
        ":math_util",
        "//tsl/platform:logging",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "math_util.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = internal_visibility(["//tensorflow/core:__pkg__"]),
)

exports_files([
    "math_util.h",
])
