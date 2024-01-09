load("//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//tsl:tsl.bzl", "set_external_visibility")
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "math_util",
    hdrs = ["math_util.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tsl_cc_test(
    name = "math_util_test",
    size = "small",
    srcs = [
        "math_util_test.cc",
    ],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

exports_files(
    [
        "math_util.h",
    ],
    visibility = ["//visibility:public"],
)
