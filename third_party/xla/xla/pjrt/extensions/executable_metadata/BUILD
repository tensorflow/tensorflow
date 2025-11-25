load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "executable_metadata_extension",
    srcs = ["executable_metadata_extension.cc"],
    hdrs = ["executable_metadata_extension.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)
