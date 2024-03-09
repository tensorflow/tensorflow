load(
    "@local_tsl//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "proto_utils",
    hdrs = ["proto_utils.h"],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
