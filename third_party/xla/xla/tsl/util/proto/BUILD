load("//xla/tsl:tsl.bzl", "if_google")
load(
    "//xla/tsl/platform:rules_cc.bzl",
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
    ] + if_google(["@com_google_protobuf//:duration_cc_proto"]),
)
