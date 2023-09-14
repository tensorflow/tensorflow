load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "proto_utils",
    hdrs = ["proto_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_headers",
    ],
)
