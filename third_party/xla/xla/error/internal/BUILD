load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":internal_users",
    ],
    licenses = ["notice"],
)

package_group(
    name = "internal_users",
    packages = [
        "//xla/error/...",
    ],
)

cc_library(
    name = "check_impl",
    hdrs = ["check_impl.h"],
    deps = [
        ":check_helper",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "check_helper",
    hdrs = ["check_helper.h"],
    deps = [
        "//xla/error:debug_me_context_util",
        "//xla/error:error_codes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)
