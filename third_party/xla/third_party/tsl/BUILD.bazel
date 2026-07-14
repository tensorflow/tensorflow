load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

license(
    name = "license",
    package_name = "tsl",
    license_kinds = ["@rules_license//licenses/spdx:Apache-2.0"],
)

cc_library(
    name = "empty",
    visibility = ["//visibility:public"],
)

# Needed to workaround https://github.com/bazelbuild/bazel/issues/21519
alias(
    name = "bazel_issue_21519",
    actual = ":empty",
    visibility = ["//visibility:public"],
)
