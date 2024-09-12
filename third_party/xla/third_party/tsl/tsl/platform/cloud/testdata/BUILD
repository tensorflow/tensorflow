# Cloud test data files.

load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "application_default_credentials",
    srcs = [
        "application_default_credentials.json",
    ],
    visibility = ["//tsl/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_credentials",
    srcs = [
        "service_account_credentials.json",
    ],
    visibility = ["//tsl/platform/cloud:__pkg__"],
)

filegroup(
    name = "service_account_public_key",
    srcs = [
        "service_account_public_key.txt",
    ],
    visibility = ["//tsl/platform/cloud:__pkg__"],
)
